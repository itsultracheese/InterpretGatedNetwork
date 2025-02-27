import argparse
import copy
import torch
from exp.experiment_classification import Experiment as ClassificationExperiment
from exp.experiment_regression import Experiment as RegressionExperiment
import random
import numpy as np

exp_dict = {
    "classification": ClassificationExperiment,
    "regression": RegressionExperiment
}


def get_args():
    parser = argparse.ArgumentParser()

    # SBM and InterpGN model hyperparameters
    parser.add_argument("--model", type=str, default='SBM', choices=['SBM', 'LTS', 'InterpGN', 'DNN'])
    parser.add_argument("--dnn_type", type=str, default='FCN', choices=['FCN', 'Transformer', 'TimesNet', 'PatchTST', 'ResNet'])
    parser.add_argument("--dataset", type=str, default="BasicMotions")
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--num_shapelet", type=int, default=10)
    parser.add_argument("--gating_value", type=float, default=None)
    parser.add_argument("--pos_weight", action="store_true")
    parser.add_argument("--sbm_cls", type=str, default='linear')
    parser.add_argument("--distance_func", type=str, default='euclidean')
    parser.add_argument("--beta_schedule", type=str, default='constant')
    parser.add_argument("--memory_efficient", action="store_true")

    # Experiment config
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=int(1e3))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--amp", action='store_false', default=True)

    # basic config
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[classification, regression]')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    # DNN model configs
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of ff layers')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # TimesNet specific
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    args = parser.parse_args()
    args.root_path = f"./data/UEA_multivariate/{args.dataset}"
    args.is_training = True
    args.data = "UEA"
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = get_args()
    exp_cls = exp_dict[args.task_name]

    random_seeds = [0, 42, 1234, 8237, 2023] if args.seed == -1 else [copy.deepcopy(args.seed)]

    for i, seed in enumerate(random_seeds):
        set_seed(seed)
        args.seed = seed

        print(f"{'=' * 5} Experiment {i} {'=' * 5} ", flush=True)
        experiment = exp_cls(args=args)
        experiment.print_args()
        print()

        if not args.test_only:
            experiment.train()
            print(f"{'=' * 5} Training Done {'=' * 5} ")
            torch.cuda.empty_cache()
            print()
        
        experiment.model.load_state_dict(torch.load(f"{experiment.checkpoint_dir}/checkpoint.pth"))
        print(f"{'=' * 5} Test {'=' * 5} ")
        cls_result = experiment.test(
            save_csv=True,
            result_dir=f"./result/{args.model}"
        )
        import pickle
        with open(f"{experiment.checkpoint_dir}/cls_result.pkl", 'wb') as f:
            pickle.dump(cls_result, f)
        print(f"Test results saved at: {experiment.checkpoint_dir}/cls_result.pkl")

        print(f"Test | Loss {cls_result.loss:.4f} | Accuracy {cls_result.accuracy:.4f}")
        print()
        torch.cuda.empty_cache()
