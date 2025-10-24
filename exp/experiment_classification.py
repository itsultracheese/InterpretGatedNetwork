import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score

from models.Shapelet import ShapeBottleneckModel, DistThresholdSBM
from models.InterpGN import InterpGN, FullyConvNetwork, Transformer, TimesNet, PatchTST, ResNet

from utils.tools import EarlyStopping, convert_to_hms, gini_coefficient
from utils.shapelet_util import ClassificationResult
from data_provider.data_factory import data_provider


def compute_beta(epoch, max_epoch, schedule='cosine'):
    if schedule == 'cosine':
        beta = 1/2 * (1 + np.cos(np.pi*epoch/max_epoch))
    elif schedule == 'linear':
        beta = 1 - epoch/max_epoch
    else:
        beta = 1
    return beta


def compute_shapelet_score(shapelet_distances, cls_weights, y_pred, y_true):
    score = shapelet_distances @ nn.functional.relu(cls_weights.T) / shapelet_distances.shape[-1]
    score_correct = score[y_pred == y_true]
    class_correct = y_true[y_pred == y_true]
    score_class = score_correct.gather(-1, class_correct.unsqueeze(1))
    return score_class.mean().item()


def get_dnn_model(configs):
    dnn_dict = {
        'FCN': FullyConvNetwork,
        'Transformer': Transformer,
        'TimesNet': TimesNet,
        'PatchTST': PatchTST,
        'ResNet': ResNet
    }
    return dnn_dict[configs.dnn_type](configs)


class Experiment(object):
    model_dict = {
        'InterpGN': InterpGN,
        'SBM': ShapeBottleneckModel,
        'LTS': DistThresholdSBM,
        'DNN': get_dnn_model
    }
    def __init__(self, args, shapelet_lengths=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8], precomputed_shapelets=False):
        self.train_data, self.train_loader = data_provider(args, flag="TRAIN")
        self.test_data, self.test_loader = data_provider(args, flag="TEST")
        self.val_data, self.val_loader = data_provider(args, flag='TEST')

        args.seq_len = max(self.train_data.max_seq_len, self.test_data.max_seq_len)
        args.pred_len = 0
        args.enc_in = self.train_data.feature_df.shape[1]
        args.num_class = len(self.train_data.class_names)
        self.epoch_stop = 0

        # Build Model
        self.args = args
        self.device = torch.device('cuda')
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = self._build_model(
            shapelet_lengths=shapelet_lengths,
            precomputed_shapelets=precomputed_shapelets
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.args.train_epochs)
        self.checkpoint_dir = "./checkpoints/{}/{}/dnn-{}_seed-{}_k-{}_div-{}_reg-{}_eps-{}_beta-{}_dfunc-{}_cls-{}".format(
            self.args.model,
            self.args.dataset,
            self.args.dnn_type,
            self.args.seed,
            self.args.num_shapelet,
            self.args.lambda_div,
            self.args.lambda_reg,
            self.args.epsilon,
            self.args.beta_schedule,
            self.args.distance_func,
            self.args.sbm_cls
        )
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def print_args(self):
        for arg in vars(self.args):
            print(f"{arg}: {getattr(self.args, arg)}")

    def _build_model(self, shapelet_lengths=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8], precomputed_shapelets=False):
        num_shapelet = [self.args.num_shapelet] * len(shapelet_lengths)

        if self.args.model == 'SBM':
            model = self.model_dict[self.args.model](
                configs=self.args,
                num_shapelet = num_shapelet,
                shapelet_len = shapelet_lengths,
                precomputed_shapelets=precomputed_shapelets
            )
        else:
            model = self.model_dict[self.args.model](
                configs=self.args,
                num_shapelet = num_shapelet,
                shapelet_len = shapelet_lengths
            )

        if self.args.multi_gpu:
            model = nn.DataParallel(model)
        return model
        
    def train(self):
        torch.set_float32_matmul_precision('medium')
        checkpoint_dir = self.checkpoint_dir

        time_start = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0)
        train_step = 0

        for epoch in range(self.args.train_epochs):
            self.model.train()

            train_loss = []
            for i, (batch_x, label, padding_mask) in enumerate(self.train_loader):
                train_step += 1
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label)
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label) + model_info.loss.mean()
                    if self.args.model in ['InterpGN']:
                        beta = compute_beta(epoch, self.args.train_epochs, self.args.beta_schedule)
                        loss += beta * nn.functional.cross_entropy(model_info.shapelet_preds, label)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                if train_step % self.args.gradient_accumulation_steps == 0:
                    if self.args.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.gradient_clip)
                    self.optimizer.step()
                    if self.args.pos_weight:
                        self.model.step()
                    self.optimizer.zero_grad()
                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            vali_loss, val_accuracy = self.validation()
            time_now = time.time()
            time_remain = (time_now - time_start) * (self.args.train_epochs - epoch) / (epoch + 1)

            if (epoch + 1) % self.args.log_interval == 0:
                print(f"Epoch {epoch}/{self.args.train_epochs} | Train Loss {train_loss:.4f} | Val Accuracy {val_accuracy:.4f} | Time Remain {convert_to_hms(time_remain)}")
            if self.args.lr_decay:
                self.scheduler.step()

            if epoch >= self.args.min_epochs:
                early_stopping(-val_accuracy, self.model, checkpoint_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                self.epoch_stop = epoch
                break
            else:
                self.epoch_stop = epoch
            sys.stdout.flush()

        best_model_path = checkpoint_dir + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def validation(self):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.val_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none')
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none') + model_info.loss.mean()
                total_loss.append(loss.flatten())

                preds.append(logits.cpu())
                trues.append(label.cpu())

        total_loss = torch.cat(total_loss, dim=0).mean().item()

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = accuracy_score(predictions, trues)

        self.model.train()
        return total_loss, accuracy


    def test(self, save_csv=True, result_dir=None):
        if not os.path.isdir(result_dir):
            try:
                os.makedirs(result_dir)
            except:
                pass
        
        @dataclass
        class Buffer:
            x_data: list = field(default_factory=list)
            trues: list = field(default_factory=list)
            preds: list = field(default_factory=list)
            shapelet_preds: list = field(default_factory=list)
            dnn_preds: list = field(default_factory=list)
            p: list = field(default_factory=list)
            d: list = field(default_factory=list)
            eta: list = field(default_factory=list)
            loss: list = field(default_factory=list)

        buffer = Buffer()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none')
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None, gating_value=self.args.gating_value)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none') + model_info.loss.mean()
                
                buffer.loss.append(loss.flatten())
                buffer.x_data.append(batch_x.cpu())
                buffer.trues.append(label.cpu())
                buffer.preds.append(logits.cpu())
                if self.args.model != 'DNN':
                    buffer.p.append(model_info.p.cpu())
                    buffer.d.append(model_info.d.cpu())
                    buffer.shapelet_preds.append(model_info.shapelet_preds.cpu())
                    if self.args.model == 'InterpGN':
                        buffer.eta.append(model_info.eta.cpu())
                        buffer.dnn_preds.append(model_info.dnn_preds.cpu())
                        
        probs = torch.nn.functional.softmax(torch.cat(buffer.preds, dim=0), dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1)  # (total_samples,) int class index for each sample
        trues = torch.cat(buffer.trues, dim=0).flatten()
        accuracy = accuracy_score(predictions.cpu().numpy(), trues.cpu().numpy())


        cls_result = ClassificationResult(
            x_data=torch.cat(buffer.x_data, dim=0).cpu(),
            trues=trues.cpu(),
            preds=predictions.cpu(),
            loss=torch.cat(buffer.loss, dim=0).mean().item(),
            accuracy=accuracy
        )

        
        if self.args.model != 'DNN':
            cls_result.p = torch.cat(buffer.p, dim=0).cpu()
            cls_result.d = torch.cat(buffer.d, dim=0).cpu()
            cls_result.shapelet_preds = torch.cat(buffer.shapelet_preds, dim=0).cpu()
            if self.args.model == 'InterpGN':
                cls_result.eta = torch.cat(buffer.eta, dim=0).cpu()
                cls_result.w = self.model.sbm.output_layer.weight.detach().cpu()
                cls_result.shapelets = self.model.sbm.get_shapelets()
            else:
                cls_result.w = self.model.output_layer.weight.detach().cpu()
                cls_result.shapelets = self.model.get_shapelets()

        if save_csv:
            summary_dict = dict()
            for arg in vars(self.args):
                if arg in [
                    'model', 'dataset', 'dnn_type', 
                    'train_epochs', 'num_shapelet', 'lambda_reg', 'lambda_div', 'epsilon', 'lr', 
                    'seed', 'pos_weight', 'beta_schedule', 'gating_value',
                    'distance_func', 'sbm_cls'
                ]:
                    summary_dict[arg] = [getattr(self.args, arg)]

            summary_dict['test_accuracy'] = accuracy
            summary_dict['epoch_stop'] = self.epoch_stop
            if self.args.model != 'DNN':
                summary_dict['eta_mean'] = cls_result.eta.mean().cpu().item() if self.args.model == 'InterpGN' else None
                summary_dict['eta_std'] = cls_result.eta.std().cpu().item() if self.args.model == 'InterpGN' else None
                summary_dict['shapelet_score'] = compute_shapelet_score(cls_result.d, cls_result.w, cls_result.preds, cls_result.trues)
                summary_dict['w_count_1'] = (cls_result.w.abs() > 1).float().sum().item()
                summary_dict['w_ratio_1'] = (cls_result.w.abs() > 1).float().mean().item()
                summary_dict['w_count_0.5'] = (cls_result.w.abs() > 0.5).float().sum().item()
                summary_dict['w_ratio_0.5'] = (cls_result.w.abs() > 0.5).float().mean().item()
                summary_dict['w_count_0.1'] = (cls_result.w.abs() > 0.1).float().sum().item()
                summary_dict['w_ratio_0.1'] = (cls_result.w.abs() > 0.1).float().mean().item()
                summary_dict['w_max'] = cls_result.w.abs().max().item()
                summary_dict['w_gini_clip'] = gini_coefficient(np.clip(cls_result.w, 0, None))
                summary_dict['w_gini_abs'] = gini_coefficient(np.abs(cls_result.w))
            
            summary_df = pd.DataFrame(summary_dict)
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            summary_df.to_csv(f"{result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv", index=False)
            print(f"Test summary saved at: {result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv")

        return cls_result.loss, cls_result, summary_df