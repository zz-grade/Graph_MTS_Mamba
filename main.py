import torch
import argparse
from datetime import datetime
import numpy as np
from models.primary_model import Base_model
from trainer.train_Graph_MTS import Trainer
from utils import _logger
import os
from importlib import import_module
from loader_data.Gnn_dataloader import data_generator


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
                    help='Dataset directory containing train.pt and test.pt. Defaults to ./data/<selected_dataset>')
parser.add_argument('--num_runs', default=1, type=int,
                    help='Number of repeated runs. Each run increments the seed by 1.')


def main(configs, args, lambda1, lambda2, lambda3, num_remain_aug1, num_remain_aug2):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    experiment_description = args.experiment_description
    method = 'GCC'
    training_mode = args.training_mode
    run_description = args.run_description
    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok = True)

    SEED = args.seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      training_mode + f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.selected_dataset}')
    logger.debug(f'Method:  {method}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug("=" * 45)


    data_path = args.data_path or os.path.join(args.home_path, "data", args.selected_dataset)
    train_dl, test_dl = data_generator(data_path, configs, args)

    logger.debug("Data loaded ...")

    model = Base_model(configs, args).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=getattr(configs, "weight_decay", 3e-4))

    print("Using device:", device)

    Trainer(model, model_optimizer, train_dl, test_dl, device, logger, configs, args)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_UEA = ['FingerMovements', 'HAR', 'ISRUC', 'ArticularyWordRecognition', 'SpokenArabicDigitsEq']

    default_lambda1 = 0.7
    default_lambda2 = 0.7
    default_lambda3 = 0.5

    args = parser.parse_args()

    config_module = import_module(f'config_files.{args.selected_dataset}_Configs')
    configs = config_module.Config()

    num_remain_aug1 = 8
    num_remain_aug2 = 2

    base_seed = args.seed
    for j in range(args.num_runs):
        args.seed = base_seed + j
        main(configs, args, default_lambda1, default_lambda2, default_lambda3, num_remain_aug1, num_remain_aug2)

