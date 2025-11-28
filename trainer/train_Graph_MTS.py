import torch.nn as nn
import torch
import numpy as np
import time


def Trainer(model, model_optimizer, train_dl, test_dl, device, logger, configs, args):
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
            accu_val = Cross_validation(model, train_dl, device)
            if accu_val > cross_accu:
                cross_accu = accu_val
                test_accu, prediction, real = Prediction(model, test_dl, device)

                print('In the {}th epoch, TESTING accuracy is {}%'.format(epoch, np.round(test_accu, 3)))
                test_accu_.append(test_accu)
                prediction_.append(prediction)
                labels.append(real)

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, model_optimizer, criterion, train_loader, device):
    model.train()
    num = int(len(train_loader) * 0.6)
    loss_ = 0
    i = 0
    for data, labels in train_loader:
        i += 1
        if i >= num:
            break
        data, labels = data.float().to(device), labels.long().to(device)
        model_optimizer.zero_grad()
        prediction = model(data)
        loss = criterion(prediction, labels)
        loss.backward()
        model_optimizer.step()
        loss_ = loss_ + loss.item()
    return loss_


def Cross_validation(model, train_loader, device):
    model.eval()
    num = int(len(train_loader) * 0.6)
    prediction_ = []
    real_ = []
    i = 0
    for data, label in train_loader:
        i += 1
        if i < num:
            continue
        data, labels = data.float().to(device), label.long().to(device)
        real_.append(label)
        prediction = model(data)
        prediction_.append(prediction.detach().cpu())
    prediction_ = torch.cat(prediction_, 0)
    real_ = torch.cat(real_, 0)
    prediction_ = torch.argmax(prediction_, -1)
    accu = accu_cal(prediction_, real_)
    return accu


def Prediction(model, test_loader, device):
    '''
    This is to predict the results for testing dataset
    :return:
    '''
    model.eval()
    prediction_ = []
    real_ = []
    for data, label in test_loader:
        data, label = data.float().to(device), label.long().to(device)
        real_.append(label)
        prediction = model(data)
        prediction_.append(prediction.detach().cpu())
    prediction_ = torch.cat(prediction_, 0)
    real_ = torch.cat(real_, 0)

    prediction_ = torch.argmax(prediction_, -1)
    accu = accu_cal(prediction_, real_)
    return accu, prediction_, real_


def accu_cal(predicted, real):
    num = predicted.size(0)
    real_num = 0
    for i in range(num):
        if predicted[i] == real[i]:
            real_num += 1
    return 100 * real_num / num
