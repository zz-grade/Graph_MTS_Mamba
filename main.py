import argparse
import importlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from datetime import datetime
import torch
from models.primary_model import Base_model
from trainer.train_Graph_MTS import Trainer
from utils import _logger, fix_randomness
from loader_data.Gnn_dataloader import data_generator, data_generator2, synthetic_data_generator


start_time = datetime.now()

parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='GCC', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='train_linear', type=str)
parser.add_argument('--selected_dataset', default='FingerMovements', type=str)
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--data_path', default=None, type=str,
                    help='Dataset directory. Defaults to <home_path>/data/<selected_dataset>')
parser.add_argument('--num_runs', default=10, type=int,
                    help='Number of repeated runs with consecutive seeds')
parser.add_argument('--smoke_test', action='store_true',
                    help='Run a one-epoch synthetic-data sanity check without dataset files')
parser.add_argument('--no_deterministic', dest='deterministic', action='store_false',
                    help='Disable deterministic PyTorch settings')
parser.set_defaults(deterministic=True)
parser.add_argument('--strict_determinism', action='store_true',
                    help='Raise an error when PyTorch detects a nondeterministic operation')
parser.add_argument('--amp', action='store_true',
                    help='Enable CUDA mixed precision. Disabled by default for reproducible runs')
parser.add_argument('--same_seed_runs', action='store_true',
                    help='Use the same seed for every run when --num_runs is greater than 1')


def apply_config_defaults(configs):
    total_nodes = max(1, configs.num_nodes * getattr(configs, "convo_time_length", 1))
    defaults = {
        "weight_decay": 3e-4,
        "mlp_hidden": getattr(configs, "dimension_token", configs.hidden_channels) * 4,
        "mpnn_layer": 1,
        "edge_num": min(10, max(1, total_nodes - 1)),
        "similar_edge": min(7, max(1, total_nodes - 1)),
        "random_edge": 0,
        "sample_num": getattr(configs, "repeat_sample", 1),
        "num_anchors": min(8, max(1, configs.num_nodes)),
    }
    for name, value in defaults.items():
        if not hasattr(configs, name):
            setattr(configs, name, value)


def configure_smoke_test(configs):
    configs.num_epoch = 1
    configs.batch_size = min(4, max(2, configs.num_classes))
    configs.batch_size_test = configs.batch_size
    configs.drop_last = False
    configs.wavelet_aug = False


def load_configs(dataset_name):
    module = importlib.import_module(f"config_files.{dataset_name}_Configs")
    return module.Config()


def main(configs, args):
    device_name = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    experiment_description = args.experiment_description
    method = 'GCC'
    training_mode = args.training_mode
    run_description = args.run_description
    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok = True)

    SEED = args.seed
    fix_randomness(SEED, deterministic=args.deterministic,
                   warn_only= args.strict_determinism)

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      training_mode + f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m')}_FingerMovements.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.selected_dataset}')
    logger.debug(f'Method:  {method}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug(f'Device:  {device}')
    logger.debug(f'Seed:    {SEED}')
    logger.debug(f'Deterministic: {args.deterministic}')
    logger.debug(f'AMP:     {args.amp}')
    logger.debug("=" * 45)


    data_path = args.data_path or os.path.join(args.home_path, "data", args.selected_dataset)
    if args.smoke_test:
        train_dl, val_dl, test_dl = synthetic_data_generator(configs, args)
    elif os.path.exists(os.path.join(data_path, "val.pt")):
        train_dl, val_dl, test_dl = data_generator2(data_path, configs, args)
    else:
        train_file = os.path.join(data_path, "train.pt")
        test_file = os.path.join(data_path, "test.pt")
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(
                f"Dataset files not found in {data_path}. Expected train.pt and test.pt, "
                "or run with --smoke_test for a synthetic sanity check."
            )
        train_dl, val_dl, test_dl = data_generator(data_path, configs, args, return_val=True)

    logger.debug("Data loaded ...")

    model = Base_model(configs, args, device).to(device)

    model_optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                        weight_decay=getattr(configs, "weight_decay", 3e-4))

    print("Using device:", device)

    Trainer(model, model_optimizer, train_dl, test_dl, device, logger, configs, args)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parser.parse_args()

    data_type = args.selected_dataset
    base_seed = args.seed
    for run_idx in range(args.num_runs):
        args.seed = base_seed if args.same_seed_runs else base_seed + run_idx
        configs = load_configs(data_type)
        apply_config_defaults(configs)
        if args.smoke_test:
            configure_smoke_test(configs)
        main(configs, args)

