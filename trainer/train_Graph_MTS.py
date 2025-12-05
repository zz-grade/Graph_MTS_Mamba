import torch.nn as nn
import torch
import numpy as np
from collections import Counter
import time


def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, logger, configs, args):
    logger.debug("Training started ....")
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    cross_accu = 0
    test_accu_ = []
    prediction_ = []
    labels = []
    for epoch in range(1, configs.num_epoch + 1):
        loss = model_train(model, model_optimizer, criterion, train_dl, device)

        if epoch % configs.show_interval == 0:
            accu_val = Cross_validation(model, train_dl, val_dl, device)
            print("cross_accu", cross_accu)
            print("accu_val", accu_val)
            # if accu_val > cross_accu:
            cross_accu = accu_val
            test_loss, test_accu, test_f1, prediction, real = Prediction(model, criterion, test_dl, device)
            scheduler.step(test_loss)
            logger.debug('In the {}th epoch, TESTING accuracy is {}%'.format(epoch, np.round(test_accu, 3)))
            logger.debug('In the {}th epoch, TESTING MacroF1 is {}%'.format(epoch, np.round(test_f1, 3)))
            test_accu_.append(test_accu)
            prediction_.append(prediction)
            labels.append(real)

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, model_optimizer, criterion, train_loader, device):
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    # num = int(len(train_loader.dataset) * 0.8)
    # print("train_loader.dataset", len(train_loader.dataset))
    loss_ = 0
    i = 0
    for data, labels in train_loader:
        i += 1
        # print("i", i)
        # if i >= num:
        #     break
        data, labels = data.float().to(device, non_blocking=True), labels.long().to(device, non_blocking=True)
        model_optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            prediction = model(data)
            # print("prediction", prediction)
            # print("labels", labels)
            loss = criterion(prediction, labels)
        scaler.scale(loss).backward()
        scaler.step(model_optimizer)
        scaler.update()
        # prediction = model(data)
        # loss = criterion(prediction, labels)
        # loss.backward()
        # model_optimizer.step()
        loss_ = loss_ + loss.item()
    return loss_


def Cross_validation(model, train_loader, val_dl, device):
    model.eval()
    # num = int(len(train_loader.dataset) * 0.8)
    prediction_ = []
    real_ = []
    i = 0
    with torch.no_grad():
        for data, label in val_dl:
            i += 1
            # if i < num:
            #     continue
            data, labels = data.float().to(device), label.long().to(device)
            real_.append(label)
            prediction = model(data)
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
            prediction = model(data)
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