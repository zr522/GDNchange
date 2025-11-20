import logging
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr


log_dir = './log'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'training_log.txt'),  # 日志文件名
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
)

# 加权MSELoss
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights  # 权重按传感器设置

    def forward(self, inputs, targets):
        loss = (inputs - targets) ** 2
        weighted_loss = loss * self.weights  # 按传感器的权重加权
        return weighted_loss.mean()

# 对比损失 (Contrastive Loss)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    device = get_device()  # 获取当前设备
    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    model.to(device)  # 确保模型在正确的设备上

    now = time.time()

    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    # 使用学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)

            loss.backward()
            optimizer.step()


            train_loss_list.append(loss.item())
            acu_loss += loss.item()

            i += 1


        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
            i_epoch, epoch,
            acu_loss/len(dataloader), acu_loss), flush=True
        )
        logging.info('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
            i_epoch, epoch,
            acu_loss/len(dataloader), acu_loss))

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)
            scheduler.step(val_loss)  # 调整学习率

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss



    return train_loss_list
