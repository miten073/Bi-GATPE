# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from pygrinder import mcar,fill_and_get_mask_torch
import os

from datasets.TimeDataset import TimeDataset

from models.GDN import GDN

from train import train
from test  import test
from evaluate import evaluate_mae,evaluate_r2,calculate_rmse,calculate_mape,evaluate_mse
from pypots.utils.metrics import calc_mae,calc_rmse

from datetime import datetime
import argparse
from pathlib import Path

import random
import matplotlib.pyplot as plt

class Main():
    def __init__(self, train_config, env_config, debug=False):
        
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        self.scaler = None
        self.dataname = 'data6'#此处修改数据集名称
        self.datasetname = 'cwt90%'#此处修改数据集名称
        self.if_save_result = True

        
        # 读取CSV文件
        file_path = "data/{}/{}.csv".format(self.datasetname,self.dataname)
        # file_path = "data/cwt_crrt_20%/{}.csv".format(self.dataname)
        df = pd.read_csv(file_path)
        dataset = self.env_config['dataset']

        # 百分之八十作为训练集，百分之二十作为测试集
        split_idx1 = int(len(df) * (1-train_config['test_ratio'])) 

        # 划分训练集和测试集
        train_data = df.iloc[:split_idx1]
        test_data = df.iloc[split_idx1:]

        # 选择要标准化的列
        selected_columns = ["vfb0", "vfb1", "vfb2", "vfb3", "vfb4", "vfb5", "label"]
        # selected_columns = ['current0', 'current1', 'current2', 'current3', 'current4', 'current5']
        self.columns = selected_columns

        # 从选定列中排除"label"列
        features_columns = [col for col in selected_columns if col != "label"]
        train_data_nolabel = train_data[selected_columns]
        test_data_nolabel = test_data[selected_columns]

        # 对选定列进行标准化
        self.scaler = StandardScaler()
        train_standardized_data = self.scaler.fit_transform(train_data_nolabel[features_columns])
        test_standardized_data = self.scaler.transform(test_data_nolabel[features_columns])

        # 将标准化后的数据转为DataFrame
        train_standardized_df = pd.DataFrame(train_standardized_data, columns=features_columns)


        # 将 "label" 列添加回去
        train_standardized_df["label"] = train_data["label"]
        test_standardized_df = pd.DataFrame(np.hstack((test_standardized_data, test_data["label"].values.reshape(-1,1))), 
                                    columns=features_columns + ["label"])

        # test_standardized_df["label"] = test_data["label"]

        feature_map = get_feature_map(dataset)#返回特征
        fc_struc = get_fc_graph_struc(dataset)#返回全连接的特征字典

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train_standardized_df.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }
       
        # 百分之八十作为训练集，百分之二十作为测试集
        split_idx2 = int(len(train_standardized_df) * (1-train_config['val_ratio'])) 

        # 划分训练集和验证
        train_data = train_standardized_df.iloc[:split_idx2]
        val_data = train_standardized_df.iloc[split_idx2:]

        # 构建训练集Dataset
        train_input = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32).permute(1, 0)
        train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float32)
        train_dataset = TimeDataset(train_input, fc_edge_index, train_labels, mode='train', config=cfg)

        # 构建验证集Dataset 
        val_input = torch.tensor(val_data.iloc[:, :-1].values, dtype=torch.float32).permute(1, 0)
        val_labels = torch.tensor(val_data.iloc[:, -1].values, dtype=torch.float32)
        val_dataset = TimeDataset(val_input, fc_edge_index, val_labels, mode='val', config=cfg)

        test_input = torch.tensor(test_standardized_df.iloc[:, :-1].values, dtype=torch.float32).permute(1, 0)
        test_labels = torch.tensor(test_standardized_df.iloc[:, -1].values, dtype=torch.float32)
        test_dataset = TimeDataset(test_input,fc_edge_index,test_labels ,mode='test', config=cfg)#返回x,y,label 

        # i = 0
        # for x, y,label,edge_index in train_dataset:
        #     if i == 2:
        #         break
        #     print(x)
        #     i += 1

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch'],
                    shuffle=True, num_workers=0)
        self.val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch'],
                    shuffle=False, num_workers=0)
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                    shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)
        
    
    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_loss_list,self.val_loss_list = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=None,
                test_dataset=None,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        


        #test（）返回真实值、预测值、标签
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)
        
        _, self.test_result = test(best_model, self.test_dataloader)#测试集测试结果，[0]是预测值，[1]是真实值
        _, self.val_result = test(best_model, self.val_dataloader)#验证集

        indicating_mask = self.test_result[3]


        self.get_score(self.test_result, self.val_result,indicating_mask)

        #用于保存数据集
        test_predicted_values = self.scaler.inverse_transform(self.test_result[0].cpu())
        test_true_values = self.scaler.inverse_transform(self.test_result[1].cpu())
        test_true_values,_ = fill_and_get_mask_torch(torch.tensor(test_true_values),0)#给缺失值补上0防止后续分类出问题


        #把label加上
        test_predicted_values = np.concatenate((test_predicted_values, self.test_result[2].cpu()), axis=1)
        test_true_values = np.concatenate((test_true_values, self.test_result[2].cpu()), axis=1)

        if self.if_save_result:
            # 确保目录存在
            output_dir = 'data/output/{}'.format(self.datasetname)        
            os.makedirs(output_dir, exist_ok=True)

            # 保存test_predicted_values到.csv文件
            predicted_file_path = os.path.join(output_dir, '{}_predicted_values.csv'.format(self.dataname))
            np.savetxt(predicted_file_path, test_predicted_values, delimiter=',', fmt='%.6f', header=','.join(self.columns), comments='')

            # 保存test_true_values到.csv文件
            true_file_path = os.path.join(output_dir, '{}_true_values.csv'.format(self.dataname))
            np.savetxt(true_file_path, test_true_values, delimiter=',', fmt='%.6f', header=','.join(self.columns), comments='')


    def get_score(self, test_result, val_result,indicating_mask):
        
        np_test_predict_result = np.array(test_result[0].cpu())
        np_test_true_result = np.array(test_result[1].cpu())
        indicating_mask = np.array(indicating_mask.cpu())
        # np_val_result = np.array(val_result)

        test_predicted_values = self.scaler.inverse_transform(np_test_predict_result)
        test_true_values = self.scaler.inverse_transform(np_test_true_result)
        test_true_values,_ = fill_and_get_mask_torch(torch.tensor(test_true_values),0)#给缺失值补上0防止后续分类出问题

        print('=========================** Result **============================\n')

        print(f'MAE: {calc_mae(test_predicted_values, np.array(test_true_values), indicating_mask)}\n')
        print(f'RMSE: {calc_rmse(test_predicted_values, np.array(test_true_values), indicating_mask)}\n')

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = f"{self.dataname}_{self.datestr}"  # 加上 self.dataname
        
        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=500)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='ett1')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-test_ratio', help='test ratio', type = float, default=0.2)
    parser.add_argument('-topk', help='topk num', type = int, default=4)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    
    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'topk': args.topk,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    
    main = Main(train_config, env_config, debug=False)
    main.run()





