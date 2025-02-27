import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
import time
import math
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

def subsample(x, max_length=1000):
    if x.shape[1] >= max_length:
        scale_factor = math.ceil(x.shape[1] / max_length)
        return x[:, ::scale_factor, :]
    else:
        return x


def compute_shapelet_score(shapelet_distances, cls_weights, y_pred, y_true):
    score = shapelet_distances @ nn.functional.relu(cls_weights.T) / shapelet_distances.shape[-1]
    score_correct = score[y_pred == y_true]
    class_correct = y_true[y_pred == y_true]
    score_class = score_correct.gather(-1, torch.from_numpy(class_correct).unsqueeze(1))
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


class CRPSLoss(nn.Module):
    def __init__(self, bin_edges):
        super(CRPSLoss, self).__init__()
        self.register_buffer("bin_edges", bin_edges)

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        cdf_pred = torch.cumsum(pred, dim=1)
        
        # Expand target to (batch_size, num_bins) for comparison with bin_edges
        target_expanded = target.unsqueeze(1)  # (batch_size, 1)
        # Create the empirical CDF: for each sample, bins where bin_edge >= target get 1, else 0.
        cdf_true = (self.bin_edges.unsqueeze(0) >= target_expanded).float()  # (batch_size, num_bins)
        
        # Compute CRPS as the mean (over batch) of the sum (over bins) of squared differences.
        loss = torch.mean(torch.sum((cdf_pred - cdf_true) ** 2, dim=1))
        return loss


class Experiment(object):
    model_dict = {
        'InterpGN': InterpGN,
        'SBM': ShapeBottleneckModel,
        'LTS': DistThresholdSBM,
        'DNN': get_dnn_model
    }
    def __init__(self, args):
        self.train_data, self.train_loader = data_provider(args, flag="TRAIN")
        self.test_data, self.test_loader = data_provider(args, flag="TEST", bin_edges=self.train_data.bin_edges)
        self.val_data, self.val_loader = data_provider(args, flag='TEST', bin_edges=self.train_data.bin_edges)

        args.seq_len = max(self.train_data.max_seq_len, self.test_data.max_seq_len)
        x, y, _ = next(iter(self.train_loader))
        # args.seq_len = max(self.train_data.max_seq_len, self.test_data.max_seq_len)
        args.seq_len = subsample(x).shape[1]
        args.pred_len = 0
        args.enc_in = self.train_data.feature_df.shape[1]
        args.num_class = len(self.train_data.class_names)
        self.epoch_stop = 0

        # Build Model
        self.args = args
        self.device = torch.device('cuda')
        self.loss_fn = CRPSLoss(torch.from_numpy(self.train_data.bin_edges)).to(self.device)
        self.model = self._build_model().to(self.device)
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

    def _build_model(self):
        shapelet_lengths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
        num_shapelet = [self.args.num_shapelet] * len(shapelet_lengths)

        model = self.model_dict[self.args.model](
            configs=self.args,
            num_shapelet = num_shapelet,
            shapelet_len = shapelet_lengths,
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
                batch_x = subsample(batch_x)
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = self.loss_fn(logits, label)
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None)
                        loss = self.loss_fn(logits, label) + model_info.loss.mean()
                    if self.args.model in ['InterpGN']:
                        beta = compute_beta(epoch, self.args.train_epochs, self.args.beta_schedule)
                        loss += beta * self.loss_fn(model_info.shapelet_preds, label)

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
            vali_loss = self.validation()
            time_now = time.time()
            time_remain = (time_now - time_start) * (self.args.train_epochs - epoch) / (epoch + 1)

            if (epoch + 1) % self.args.log_interval == 0:
                print(f"Epoch {epoch}/{self.args.train_epochs} | Train Loss {train_loss:.4f} | Val Loss {vali_loss:.4f} | Time Remain {convert_to_hms(time_remain)}")
            if self.args.lr_decay:
                self.scheduler.step()

            if epoch >= self.args.min_epochs:
                early_stopping(vali_loss, self.model, checkpoint_dir)
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
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.val_loader):
                batch_x = subsample(batch_x)
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = self.loss_fn(logits, label)
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None)
                        loss = self.loss_fn(logits, label) + model_info.loss.mean()
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


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
                batch_x = subsample(batch_x)
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = self.loss_fn(logits, label)
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None, gating_value=self.args.gating_value)
                        loss = self.loss_fn(logits, label) + model_info.loss.mean()
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

        total_loss = np.average(buffer.loss)
        raw = torch.cat(buffer.x_data, dim=0)
        preds = torch.cat(buffer.preds, dim=0)
        labels = torch.cat(buffer.trues, dim=0)
        
        if self.args.model != 'DNN':
            predicates = torch.cat(buffer.p, dim=0)
            distances = torch.cat(buffer.d, dim=0)
            sbm_preds = torch.cat(buffer.shapelet_preds, dim=0)
            eta = torch.cat(buffer.eta, dim=0) if self.args.model == 'InterpGN' else None
            eta_mean = eta.mean().cpu().item() if self.args.model == 'InterpGN' else None
            eta_std = eta.std().cpu().item() if self.args.model == 'InterpGN' else None

            w = self.model.sbm.output_layer.weight.detach().cpu() if self.args.model == 'InterpGN' else self.model.output_layer.weight.detach().cpu() 
            w_sum_10 = (w.abs() > 1).float().sum().item()
            w_mean_10 = (w.abs() > 1).float().mean().item()
            w_sum_5 = (w.abs() > 0.5).float().sum().item()
            w_mean_5 = (w.abs() > 0.5).float().mean().item()
            w_sum_1 = (w.abs() > 0.1).float().sum().item()
            w_mean_1 = (w.abs() > 0.1).float().mean().item()
            w_max = w.abs().max().item()
            w_gini_clip = gini_coefficient(np.clip(w, 0, None))
            w_gini_abs = gini_coefficient(np.abs(w))

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

            summary_dict['test_loss'] = total_loss
            summary_dict['epoch_stop'] = self.epoch_stop
            if self.args.model != 'DNN':
                summary_dict['eta_mean'] = eta_mean
                summary_dict['eta_std'] = eta_std
                summary_dict['w_sum_10'] = w_sum_10
                summary_dict['w_mean_10'] = w_mean_10
                summary_dict['w_sum_5'] = w_sum_5
                summary_dict['w_mean_5'] = w_mean_5
                summary_dict['w_sum_1'] = w_sum_1
                summary_dict['w_mean_1'] = w_mean_1
                summary_dict['w_max'] = w_max
                summary_dict['w_gini_clip'] = w_gini_clip
                summary_dict['w_gini_abs'] = w_gini_abs
            
            summary_df = pd.DataFrame(summary_dict)
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            summary_df.to_csv(f"{result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv", index=False)
            print(f"Test summary saved at: {result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv")

        if self.args.model != 'DNN':
            df = {
                'x': raw.float().cpu().numpy(),
                'pred': preds.float().cpu().numpy(),
                'target': labels.float().cpu().numpy(),
                'predicate': predicates.float().cpu().numpy(),
                'w': w.float().cpu().numpy(),
                'shapelets': self.model.get_shapelets() if self.args.model == 'SBM' else self.model.sbm.get_shapelets(),
                'eta': eta.float().cpu().numpy() if self.args.model == 'InterpGN' else None,
                'sbm_pred': sbm_preds.float().cpu().numpy() if self.args.model == 'InterpGN' else None
            }
        else:
            df = {
                'x': raw.float().cpu().numpy(),
                'pred': preds.float().cpu().numpy(),
                'target': labels.float().cpu().numpy()
            }

        return total_loss, None, df