import torch.nn as nn
import torch


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, test_dl, device, logger, config, experiment_log_dir, training_mode, lambda1, lambda2, lambda3,
            num_remain_aug1, num_remain_aug2):
    logger.debug("Training started ....")
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    loss = []
    for epoch in range(1, config.num_epoch + 1):
        train_loss, train_acc, train_loss_details = "model_train"