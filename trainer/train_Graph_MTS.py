import os
from os import write

import torch.nn as nn
import torch
import numpy as np
from collections import Counter

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time


def Trainer(model, model_optimizer, train_dl, test_dl, device, logger, configs, args):
    # writer = SummaryWriter("runs/mem")
    # global_step = 0
    print(datetime.now())
    logger.debug("Training started ....")
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    cross_accu = 0
    test_accu_ = []
    prediction_ = []
    labels = []

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, configs.num_epoch + 1):
        # train_sampler.set_epoch(epoch)
        # test_sampler.set_epoch(epoch)
        loss = model_train(model, model_optimizer, criterion, train_dl, device, epoch, scaler, use_amp, configs.epsilon)

        if epoch % configs.show_interval == 0:
            accu_val = Cross_validation(model, train_dl, device)
            rank = int(os.environ.get("RANK", "0"))
            is_main = (rank == 0)
            if is_main:
                print(datetime.now(), "cross_accu", cross_accu)
                print(datetime.now(), "accu_val", accu_val)
            # if accu_val > cross_accu:
            cross_accu = accu_val
            test_loss, test_accu, test_f1, prediction, real = Prediction(model, criterion, test_dl, device)
            scheduler.step(test_loss)
            if is_main:
                logger.debug('{} In the {}th epoch, TESTING accuracy is {}%'.format(datetime.now(), epoch, np.round(test_accu, 3)))
                logger.debug('{} In the {}th epoch, TESTING MacroF1 is {}%'.format(datetime.now(), epoch, np.round(test_f1, 3)))
            test_accu_.append(test_accu)
            prediction_.append(prediction)
            labels.append(real)

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, model_optimizer, criterion, train_loader, device, epoch, scaler, use_amp, epsilon):
    model.train()
    # num = int(len(train_loader.dataset) * 0.8)
    # print("train_loader.dataset", len(train_loader.dataset))
    loss_ = 0
    i = 0
    for data, labels in train_loader:


        i += 1
        # if i>=100:
        #     break
        # print("i", i)
        # if i >= num:
        #     break
        data, labels = data.float().to(device, non_blocking=True), labels.long().to(device, non_blocking=True)
        model_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            prediction, loss_xx = model(data, labels=labels)
            # print("prediction", prediction)
            # print("labels", labels)
            loss_cls = criterion(prediction, labels)
            loss = loss_cls + loss_xx

        # print(datetime.now(), "反向传递开始")
        scaler.scale(loss).backward()
        # if epoch>=30:
        #     # 先反缩放
        # scaler.unscale_(model_optimizer)
        # fim_l1, scale = apply_fic_constraint(model, epsilon)
        # print(f"fim_l1={fim_l1:.6f}, scale={scale:.6f}")
        # print(datetime.now(), "反向传递完成")
        scaler.step(model_optimizer)
        scaler.update()
        # prediction = model(data)
        # loss = criterion(prediction, labels)
        # loss.backward()
        # model_optimizer.step()
        loss_ = loss_ + loss.item()

        # if global_step % 3 == 0:  # 每10步记一次，防止日志太大
        #     writer.add_scalar("mem/allocated_MB", torch.cuda.memory_allocated() / 1024 ** 2, global_step)
        #     writer.add_scalar("mem/reserved_MB", torch.cuda.memory_reserved() / 1024 ** 2, global_step)
        #     writer.add_scalar("mem/peak_allocated_MB", torch.cuda.max_memory_allocated() / 1024 ** 2, global_step)
        #
        # global_step += 1
    return loss_


def Cross_validation(model, train_loader, device):
    model.eval()
    # num = int(len(train_loader.dataset) * 0.8)
    prediction_ = []
    real_ = []
    i = 0
    with torch.no_grad():
        for data, label in train_loader:
            i += 1
            # if i < num:
            #     continue
            data, labels = data.float().to(device), label.long().to(device)
            real_.append(label)
            prediction, _ = model(data)
            prediction_.append(prediction.detach().cpu())
        prediction_ = torch.cat(prediction_, 0)
        real_ = torch.cat(real_, 0)
        prediction_ = torch.argmax(prediction_, -1)
        # print("prediction_", prediction_)
        # print("real_", real_)
        accu = accu_cal(prediction_, real_)
    return accu


def Prediction(model, criterion, test_loader, device):
    '''
    This is to predict the results for testing dataset
    :return:
    '''
    model.eval()
    prediction_ = []
    real_ = []
    loss_ = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.float().to(device), label.long().to(device)
            real_.append(label)
            prediction, _ = model(data)
            loss = criterion(prediction, label)
            prediction_.append(prediction.detach().cpu())
            loss_ = loss_ + loss.item()
        prediction_ = torch.cat(prediction_, 0)
        real_ = torch.cat(real_, 0)
        prediction_ = torch.argmax(prediction_, -1)
        accu = accu_cal(prediction_, real_)
        mf1 = accu_F1(prediction_, real_)
    return loss_, accu, mf1, prediction_, real_


def accu_cal(predicted, real):
    num = predicted.size(0)
    real_num = 0
    for i in range(num):
        if predicted[i] == real[i]:
            real_num += 1
    return 100 * real_num / num


def accu_F1(predicted, real):
    if torch.is_tensor(predicted):

        predicted = predicted.detach().cpu().tolist()
    if torch.is_tensor(real):
        real = real.detach().cpu().tolist()
    labels = sorted(set(predicted) | set(real))
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for p, r in zip(predicted, real):
        if p == r:
            tp[r] += 1
        else:
            fp[p] += 1
            fn[r] += 1

    f1_per_class = {}
    for c in labels:
        prec = safe_div(tp[c], tp[c] + fp[c])
        rec = safe_div(tp[c], tp[c] + fn[c])
        f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) != 0 else 0.0
        f1_per_class[c] = f1
    return sum(f1_per_class.values()) / len(labels) * 100 if labels else 0.0

def safe_div(a, b):
    return a / b if b != 0 else 0.0


@torch.no_grad()
def apply_fic_constraint(model, epsilon: float, eps: float = 1e-12):
    """
    对 model 当前已经反传得到的梯度施加 FIC 约束。

    论文对应：
        diag(F) ≈ g ∘ g
        ||F||_1 = sum(g^2)
        if ||F||_1 >= epsilon:
            g <- sqrt(epsilon / ||F||_1) * g

    参数:
        model:   你的模型
        epsilon: FIC 阈值，对应论文里的 ε
        eps:     数值稳定项
    返回:
        fim_l1:  当前的 ||F||_1
        scale:   实际使用的缩放系数
    """
    fim_l1 = None

    # 1) 累加所有参数梯度平方和：||F||_1
    total = 0.0
    device = None

    for p in model.parameters():
        if p.grad is None:
            continue
        if device is None:
            device = p.grad.device
        g = p.grad
        total = total + torch.sum(g * g)

    if device is None:
        # 没有任何梯度
        return 0.0, 1.0

    fim_l1 = total

    # 2) 判断是否超过阈值
    if fim_l1 >= epsilon:
        scale = torch.sqrt(torch.tensor(epsilon, device=device, dtype=fim_l1.dtype) / (fim_l1 + eps))

        # 3) 整体缩放所有梯度
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
    else:
        scale = torch.tensor(1.0, device=device, dtype=fim_l1.dtype)

    return fim_l1.item(), scale.item()


