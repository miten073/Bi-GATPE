import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score
from pypots.utils.metrics import calc_mae,calc_mse
from pygrinder import mcar,fill_and_get_mask_torch

from util.data import *
from util.preprocess import *



#改版后的test，修改loss的计算方式
def test(model, dataloader):
    device = get_device()
    loss_list = []
    all_predictions = []
    all_labels = []
    now = time.time()
    data_len = len(dataloader)
    model.eval()
    i = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, labels, edge_index in dataloader:
            x, labels, edge_index = [item.to(device).float() for item in [x, labels, edge_index]]

            perdict = model(x, edge_index)

            # 计算损失
            loss = F.cross_entropy(perdict, labels.long())
            total_loss += loss.item()
            loss_list.append(loss.item())

            # 计算概率
            probabilities = F.softmax(perdict, dim=1)

            # 获取预测结果
            _, predictions = probabilities.max(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            i += 1
            if i % 10000 == 1 and i > 1:
                print(timeSincePlus(now, i / data_len))

    avg_loss = sum(loss_list) / len(loss_list)

    return {
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels
    }
