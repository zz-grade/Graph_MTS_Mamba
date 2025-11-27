import torch
import torch.nn.functional as F
import argparse
from datetime import datetime

from models.some_model import Base_model
from utils import _logger, set_requires_grad
from huggingface_hub.utils import experimental
from torch_geometric.datasets import KarateClub
import os

from graph_mamba.GraphMambaNet import GraphMambaNet
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
parser.add_argument('--selected_dataset', default='HAR', type=str)
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')


def main(configs, args,lambda1, lambda2, lambda3,num_remain_aug1, num_remain_aug2):
    device = torch.device(args.device)
    experiment_description = args.experiment_description
    method = 'GCC'
    training_mode = args.training_mode
    run_description = args.run_description
    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok = True)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      training_mode + f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {method}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug("=" * 45)


    data_path = f"./data/{data_type}"
    train_dl, test_dl = data_generator("/data/user_zhangzhe/data/FingerMovements", configs, args)

    logger.debug("Data loaded ...")

    model = Base_model(configs, args).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 载入 KarateClub 数据集（一个图，34 个节点）:contentReference[oaicite:4]{index=4}
    dataset = KarateClub()
    data = dataset[0].to(device)

    # 单图场景手动加上 batch
    if not hasattr(data, "batch"):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    model = GraphMambaNet(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(data)          # [N, num_classes]
        loss = F.cross_entropy(out, data.y)  # 这里偷懒，直接全节点都当作训练样本
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            pred = out.argmax(dim=-1)
            acc = (pred == data.y).float().mean().item()
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Acc {acc:.4f}")





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_UEA = ['FingerMovements', 'HAR', 'ISRUC', 'ArticularyWordRecognition', 'SpokenArabicDigitsEq']

    default_lambda1 = 0.7
    default_lambda2 = 0.7
    default_lambda3 = 0.5

    args = parser.parse_args()

    args.selected_dataset = 'FingerMovements'
    data_type = args.selected_dataset
    exec(f'from config_files.{data_type}_Configs import Config as Configs')
    configs = Configs()

    num_remain_aug1 = 8
    num_remain_aug2 = 2

    for j in range(10):
        args.training_mode = 'self_supervised'
        main(configs, args, default_lambda1, default_lambda2, default_lambda3, num_remain_aug1, num_remain_aug2)
        args.training_mode = 'train_linear'
        main(configs, args, default_lambda1, default_lambda2, default_lambda3, num_remain_aug1, num_remain_aug2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
