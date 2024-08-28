import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test4classify import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
import os
from pypots.utils.metrics import calc_mae
from pygrinder import mcar,fill_and_get_mask_torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None,if_sharing=False):
    seed = config['seed']

    if if_sharing:
        # 获取未被冻结的参数
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        # 使用未被冻结的参数初始化优化器
        optimizer = torch.optim.RMSprop(params=params_to_optimize, lr=0.001, eps=1e-08)
    else:
        # 不共享参数使用这个
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.001, eps=1e-08)



    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['decay'])

    now = time.time()

    train_loss_list = []
    val_loss_list = []

    device = get_device()

    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 20
    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader
    total_train_time = 0

    for i_epoch in range(epoch):
        epoch_train_loss = 0
        model.train()
        start_epoch_time = time.time()

        for x, labels, edge_index in dataloader:
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()

            perdiction = model(x, edge_index)

            loss = F.cross_entropy(perdiction, labels.long())

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(dataloader)  # 计算本epoch的平均训练损失
        train_loss_list.append(epoch_train_loss)  # 将平均训练损失添加到列表中

        epoch_time = time.time() - start_epoch_time
        total_train_time += epoch_time
        print('Epoch {} completed in {:.2f} seconds'.format(i_epoch, epoch_time))

        print('epoch ({} / {}) (Train Loss:{:.8f})'.format(
            i_epoch, epoch, epoch_train_loss), flush=True
        )

        # use val dataset to judge
        if val_dataloader is not None:
            val_result  = test(model, val_dataloader)
            val_loss = val_result['loss']
            val_loss_list.append(val_loss)  # 将验证损失添加到 val_loss_list 中
            print(f'Validation Loss: {val_loss:.6f}')

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            #开关early stop
            # if stop_improve_count >= early_stop_win:
            #     break

        else:
            if epoch_train_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = epoch_train_loss

    print('Total training time: {:.2f} seconds'.format(total_train_time))

    return train_loss_list, val_loss_list  # 返回两个列表