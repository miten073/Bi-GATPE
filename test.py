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

from pypots.utils.metrics import calc_mae,calc_mse
from pygrinder import mcar,fill_and_get_mask_torch

from util.data import *
from util.preprocess import *



#改版后的test，修改loss的计算方式
def test(model, dataloader):
    # test
    # loss_func = nn.MSELoss(reduction='mean')
    loss_func = nn.L1Loss(reduction='mean')#mae
    device = get_device()

    test_loss_list = []
    now = time.time()

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []
    t_test_miss_list = []

    test_len = len(dataloader)


    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():
            output, _, missing_mask = model(x, edge_index,mode='test')
            gdn_fusion,encoder_output,combat,encoder_imputed = output[0], output[1], output[2],output[3]
            num_node = x.shape[1]

            x_input =x.permute(0,2,1).reshape(-1,num_node)#(B,N,L)=>(B*L,N)  
            fill_x,x_mask = fill_and_get_mask_torch(x_input,0)

            test_true = y.permute(0,2,1).reshape(-1,num_node)#(B,N,L)=>(B*L,N)  
            test_true_cal,true_mask = fill_and_get_mask_torch(test_true,0)

            test_labels = labels.reshape(-1,1)#(B,L,1)=>(B*L,1)  

            indicating_mask = (true_mask - x_mask).to(torch.float32)

            loss = calc_mae(encoder_imputed, test_true_cal, indicating_mask)

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = encoder_imputed
                t_test_ground_list = test_true
                t_test_labels_list = test_labels
                # t_test_miss_list = miss_ori
                t_test_indicating = indicating_mask
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, encoder_imputed), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, test_true), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, test_labels), dim=0)
                # t_test_miss_list = torch.cat((t_test_miss_list, miss_ori), dim=0)
                t_test_indicating = torch.cat((t_test_indicating, indicating_mask), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))


    avg_loss = sum(test_loss_list)/len(test_loss_list)

    # return avg_loss, [t_test_predicted_list,t_test_ground_list,t_test_labels_list,t_test_miss_list,t_test_mask]
    return avg_loss, [t_test_predicted_list,t_test_ground_list,t_test_labels_list,t_test_indicating]

