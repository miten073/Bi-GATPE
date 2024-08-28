import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from pygrinder import mcar,fill_and_get_mask_torch


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, labels, mode='train', config = None):
        self.raw_data = raw_data
        self.label = labels
        self.config = config
        self.edge_index = edge_index
        self.mode = mode


        # to tensor
        data = torch.tensor(raw_data).double()
        labels = torch.tensor(labels).double()

        self.x,self.labels = self.process(data, labels)
    

    #返回数据集中样本的数量。
    def __len__(self):
        return len(self.x)

    #数据处理函数。
    # 接收数据张量 data 和标签张量 labels，将数据进行处理并返回处理后的特征张量 x、目标张量 y 和标签张量 labels。
    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []
        slide_win = self.config['slide_win']
        slide_stride = self.config['slide_stride']
        node_num, total_time_len = data.shape
        data_mcar = mcar(data,0.1)
        rang = range(slide_win, total_time_len, slide_stride)

        for i in rang:
            ft = data[:, i-slide_win:i]
            labels_slice = labels[i-slide_win]
            x_arr.append(ft)
            labels_arr.append(labels_slice)

        x = torch.stack(x_arr).contiguous()
        labels = torch.stack(labels_arr).contiguous()
        return x, labels

    #通过索引获取数据集中的样本。
    # 接收索引 idx，返回特征张量 feature、目标张量 y、标签 label 和边缘索引 edge_index。
    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, label, edge_index





