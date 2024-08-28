# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset,DataLoader, Subset,ConcatDataset

# from torch.utils.tensorboard import SummaryWriter   

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
import os

from datasets.TimeDataset4classify import TimeDataset

from models.GDN4classify import GDN as gdn4classify
from models.GDN import GDN as gdn4imputation

from pygrinder import mcar

from train4classify import train
from test4classify  import test
from evaluate import evaluate_mae,evaluate_r2,calculate_rmse,calculate_mape,evaluate_mse
from pypots.utils.metrics import calc_mae,calc_rmse

from datetime import datetime
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import random

class MixedDataset(Dataset):
    def __init__(self, x, labels, edge_index):
        self.x = x
        self.labels = labels
        self.edge_index = edge_index

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx], self.edge_index


class Main():
    def __init__(self, train_config, env_config, debug=False):
        
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        self.scaler = None
        
        dataset_name = self.env_config['dataset']
        self.dataname = 'GDN90%'
        self.datatype = 'predicted_values'#true_values or predicted_values
        self.Parameter_Sharing = False #是否使用参数共享 True or False
        

        #读取滑窗信息
        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        # 读取六个CSV文件
        csv_files = ['data/output/{}/data{}_{}.csv'.format(self.dataname, i ,self.datatype) for i in range(7)]
        datasets = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            #只取一半
            half_length = len(df)//10
            df = df.iloc[:half_length]

            datasets.append(df)

        #从文件读取全连接图
        feature_map = get_feature_map(dataset_name)#返回特征
        fc_struc = get_fc_graph_struc(dataset_name)#返回全连接的特征字典
        self.feature_map = feature_map

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(df.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        #将每个CSV按固定比例和随机种子分割为训练集和测试集
        train_datasets = []
        test_datasets = []
        for dataset in datasets:
            train_size = int((1-train_config['test_ratio'])* len(dataset))  # 假设训练集占80%
            train_dataset = dataset.iloc[:train_size]
            test_dataset = dataset.drop(train_dataset.index)
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

        #标准化所有训练集数据,并应用于训练集和测试集
        all_train_data = pd.concat(train_datasets, ignore_index=True)
        self.scaler = StandardScaler()
        self.scaler.fit(all_train_data.iloc[:, :-1])  # 不标准化标签列

        normalized_train_datasets = []
        normalized_test_datasets = []
        for train_dataset, test_dataset in zip(train_datasets, test_datasets):
            train_data = train_dataset.iloc[:, :-1]
            train_labels = train_dataset.iloc[:, -1]
            test_data = test_dataset.iloc[:, :-1]
            test_labels = test_dataset.iloc[:, -1]

            normalized_train_data = self.scaler.transform(train_data)
            normalized_test_data = self.scaler.transform(test_data)

            normalized_train_dataset = pd.DataFrame(normalized_train_data, columns=train_data.columns)
            normalized_train_dataset['label'] = train_labels.values

            normalized_test_dataset = pd.DataFrame(normalized_test_data, columns=test_data.columns)
            normalized_test_dataset['label'] = test_labels.values #不加values会报错

            normalized_train_datasets.append(normalized_train_dataset)
            normalized_test_datasets.append(normalized_test_dataset)

        # 组合所有训练数据集
        all_train_x = []
        all_train_labels = []
        all_val_x = []
        all_val_labels = []
        all_test_x = []
        all_test_labels = []

        for train_dataset, test_dataset in zip(normalized_train_datasets, normalized_test_datasets):
            train_size = len(train_dataset)
            val_size = int(train_config['val_ratio'] * train_size)

            train_data = torch.tensor(train_dataset.iloc[:, :-1].values, dtype=torch.float32).permute(1, 0)
            train_labels = torch.tensor(train_dataset.iloc[:, -1].values, dtype=torch.float32)
            ts_train_dataset = TimeDataset(train_data, fc_edge_index, train_labels, config=cfg)

            val_data = torch.tensor(train_dataset.iloc[-val_size:, :-1].values, dtype=torch.float32).permute(1, 0)
            val_labels = torch.tensor(train_dataset.iloc[-val_size:, -1].values, dtype=torch.float32)
            ts_val_dataset = TimeDataset(val_data, fc_edge_index, val_labels, config=cfg)

            test_data = torch.tensor(test_dataset.iloc[:, :-1].values, dtype=torch.float32).permute(1, 0)
            test_labels = torch.tensor(test_dataset.iloc[:, -1].values, dtype=torch.float32)
            ts_test_dataset = TimeDataset(test_data, fc_edge_index, test_labels, config=cfg)

            all_train_x.append(ts_train_dataset.x)
            all_train_labels.append(ts_train_dataset.labels)
            all_val_x.append(ts_val_dataset.x)
            all_val_labels.append(ts_val_dataset.labels)
            all_test_x.append(ts_test_dataset.x)
            all_test_labels.append(ts_test_dataset.labels)

        # 拼接所有训练数据
        all_train_x = torch.cat(all_train_x, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)

        # 拼接所有验证数据
        all_val_x = torch.cat(all_val_x, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)

        # 拼接所有测试数据
        all_test_x = torch.cat(all_test_x, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)

        # 创建自定义数据集
        train_dataset = MixedDataset(all_train_x, all_train_labels, fc_edge_index)
        val_dataset = MixedDataset(all_val_x, all_val_labels, fc_edge_index)
        test_dataset = MixedDataset(all_test_x, all_test_labels, fc_edge_index)

        # 使用 DataLoader 加载数据集,打乱时间窗口顺序
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=train_config['batch'], shuffle=True)


        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.test_dataloader = test_loader



        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        if self.Parameter_Sharing:
            #若使用参数共享就使用这些代码
            self.classification_model = gdn4classify(edge_index_sets, len(feature_map), 
                    dim=train_config['dim'], 
                    input_dim=train_config['slide_win'],
                    out_layer_num=train_config['out_layer_num'],
                    out_layer_inter_dim=train_config['out_layer_inter_dim'],
                    topk=train_config['topk']
                ).to(self.device)

            self.imputation_model = gdn4imputation(edge_index_sets, len(feature_map), 
                    dim=train_config['dim'], 
                    input_dim=train_config['slide_win'],
                    out_layer_num=train_config['out_layer_num'],
                    out_layer_inter_dim=train_config['out_layer_inter_dim'],
                    topk=train_config['topk']
                ).to(self.device)
            
            #获取已经训练好的最佳补全模型，data6中的
            #这些是20%缺失用的
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_05|22-12:36:47.pt'))
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_05|22-12:36:47.pt'))
            #这些是10%缺失用的
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_07|02-22:19:44.pt'))
            #这些是30%缺失用的
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_07|03-15:48:40.pt'))
            # #这些是40%缺失用的
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_07|09-16:04:26.pt'))
            # #这些是50%缺失用的
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_07|11-17:44:36.pt'))
            # #这些是60%缺失用的
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data5_07|31-15:59:50.pt'))            
            # #这些是70%缺失用的
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_08|01-18:04:56.pt'))
            # #这些是80%缺失用的
            # self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_08|01-15:21:24.pt'))
            #这些是90%缺失用的
            self.imputation_model.load_state_dict(torch.load('pretrained/cwt2/best_data6_08|01-15:21:24.pt'))



            # 获取imputation_model中的共享层参数
            imputation_embedding = self.imputation_model.embedding.state_dict()
            imputation_embedding_b = self.imputation_model.embedding_b.state_dict()
            imputation_encoder_layers = [layer.state_dict() for layer in self.imputation_model.encoder_layers]
            imputation_gnn_layers = [layer.state_dict() for layer in self.imputation_model.gnn_layers]

            # 加载参数到classification_model中对应的层
            self.classification_model.embedding.load_state_dict(imputation_embedding)
            self.classification_model.embedding_b.load_state_dict(imputation_embedding_b)
            for layer, state_dict in zip(self.classification_model.encoder_layers, imputation_encoder_layers):
                layer.load_state_dict(state_dict)
            for layer, state_dict in zip(self.classification_model.gnn_layers, imputation_gnn_layers):
                layer.load_state_dict(state_dict)

            #冻结这些共享层的梯度
            for param in self.classification_model.embedding.parameters():
                param.requires_grad = False
            for param in self.classification_model.embedding_b.parameters():
                param.requires_grad = False
            for layer in self.classification_model.encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            for layer in self.classification_model.gnn_layers:
                for param in layer.parameters():
                    param.requires_grad = False

            #将classification_model设置为模型
            self.model = self.classification_model
        else:
            #不参数共享就用这个
            self.model =gdn4classify(edge_index_sets, len(feature_map), 
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
                dataset_name=self.env_config['dataset'],
                if_sharing = self.Parameter_Sharing
            )
        


        #test（）返回真实值、预测值、标签
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        
        self.test_result = test(best_model, self.test_dataloader)#测试集测试结果，[0]是预测值，[1]是真实值
        self.val_result = test(best_model, self.val_dataloader)#验证集

        test_loss = self.test_result['loss']
        test_predictions = self.test_result['predictions']
        test_labels = self.test_result['labels']

        # 计算其他评估指标
        test_accuracy = accuracy_score(test_labels, test_predictions)

        print('test_accuracy:',test_accuracy)

        # 创建目录
        save_dir = 'loss_curve'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 创建一个 DataFrame
        df = pd.DataFrame({
            'Train Loss': self.train_loss_list,
            'Validation Loss': self.val_loss_list
        })

        # 保存为 Excel 文件
        df.to_excel(os.path.join(save_dir, 'loss_lists{}_{}_{}.xlsx'.format(self.dataname,self.datatype,self.Parameter_Sharing)), index=False)

        # # 创建pic/loss_curve/文件夹(如果不存在)
        # os.makedirs('pic/loss_curve', exist_ok=True)

        # # 绘制训练损失和验证损失曲线
        # fig, ax = plt.subplots(figsize=(10, 6))

        # ax.plot(range(len(self.train_loss_list)), self.train_loss_list, label='Train Loss')
        # ax.plot(range(len(self.val_loss_list)), self.val_loss_list, label='Validation Loss')

        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Loss')
        # ax.set_title('')
        # ax.legend()

        # # 保存图像
        # fig.savefig('pic/loss_curve/{}_{}_{}.png'.format(self.dataname,self.datatype,self.Parameter_Sharing), dpi=300, bbox_inches='tight')

        # # 显示图像
        # plt.show()
    

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          
        
        paths = [
            f'./classify_pretrained/{dir_path}/best_{datestr}.pt',
            f'./classify_results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths
        # 自定义数据集



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





