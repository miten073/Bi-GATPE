import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import util
from util.time import *
from util.env import *
from util.tool4GDN import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F
from .graph_layer import GraphLayer
from pygrinder import mcar,fill_and_get_mask_torch
from scipy.spatial.distance import pdist, squareform


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()



class SAITS(nn.Module):
    def __init__(self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, **kwargs):
        super().__init__()
        self.input_with_mask = kwargs["input_with_mask"]
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        
        self.layer_stack_for_first_block = nn.ModuleList(
            [
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(kwargs["n_group_inner_layers"])
            ]
        )
        self.n_group_inner_layers = kwargs["n_group_inner_layers"]
        
        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)

    def forward(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(self.position_enc(input_X_for_first))

        for encoder_layer in self.layer_stack_for_first_block:
            for _ in range(self.n_group_inner_layers):
                enc_output, attn_weights = encoder_layer(enc_output)
        X_tilde_1 = self.reduce_dim_z(enc_output)
        return X_tilde_1,attn_weights

class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        # out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)

#node_num就是特征的个数 嵌入维度为自定义
class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets
        device = get_device()

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)#前向嵌入
        self.embedding_b = nn.Embedding(node_num, embed_dim)#后向嵌入

        edge_set_num = len(edge_index_sets) 
        layer_num = 2 #控制图神经层个数
        
        kwargs = {"diagonal_attention_mask": True,
                    "device": "cuda"}
        self.encoder_layers = nn.ModuleList([SAITS(
                            d_time=input_dim, 
                            d_feature=node_num,
                            d_model=128,
                            d_inner=256,
                            n_head=2,
                            d_k=64,
                            d_v=64,
                            dropout=0.1,
                            input_with_mask=True,
                            n_group_inner_layers=3,
                            **kwargs
                            )
        ])

        # 融合
        self.weight_combine = nn.Linear(node_num + input_dim, node_num)


        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(layer_num)
        ])

        self.leaky_relu = nn.LeakyReLU()

        self.topk = topk

        self.gdn_fclayer = nn.Linear(dim*2, input_dim)

        self.init_params()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {total_params}')
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.embedding_b.weight, a=math.sqrt(5))

    
    # 调用GDN次数=train次数+test次数+val次数，由于tra`in中包含了一次val，所以总次数=train+2*val+test
    def forward(self, data, org_edge_index,mode):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets
        device = data.device

        x = x.to(device)
        batch_num, node_num, all_feature = x.shape #x的形状是(batch_size, node_num, slide_win)/（B,N,L）

        rate = 0.2#模拟缺失率
        X_ori = x.clone().detach().permute(0, 2, 1) #(B,N,L)=>(B,L,N)
        if mode == 'train':     #训练时随机丢失20%来模拟
            X = mcar(X_ori, rate)#(B,L,N)
        else:    #测试和验证的时候不用缺失，直接补全
            X = X_ori.clone().detach()#(B,L,N)
        X, missing_mask = fill_and_get_mask_torch(X,0)#observed values are set to 1, missing values are set to 0
        X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori,0)#ori observed values are set to 1, missing values are set to 0
        indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)#only indicating artificial missing values are set to 1, observed values are set to 0


        x_f = X.permute(0, 2, 1)#(B,L,N)->(B,N,L)方便输入GDN
        x_b = torch.flip(x_f, dims=(-1,))
        x_f = x_f.contiguous().reshape(-1, all_feature)#(B,N,L)->(B*N,L)
        x_b = x_b.contiguous().reshape(-1, all_feature)#(B,N,L)->(B*N,L)

        node_embeddings = self.embedding(torch.arange(node_num).to(device))
        node_embeddings_b = self.embedding_b(torch.arange(node_num).to(device))


        gated_edge_index_a = compute_edge_index(node_embeddings, self.topk).to(device)
        gated_edge_index_b = compute_edge_index(node_embeddings_b, self.topk).to(device)

      
        batch_gated_edge_index = get_batch_edge_index(gated_edge_index_a, batch_num, node_num).to(device)#通过余弦相似度计算得来的图
        batch_gated_edge_index_b = get_batch_edge_index(gated_edge_index_b, batch_num, node_num).to(device)#通过余弦相似度计算得来的图


        batch_node_embeddings = node_embeddings.repeat(batch_num, 1)
        batch_node_embeddings_b = node_embeddings_b.repeat(batch_num, 1)


        x_f_in = x_f.clone().detach()#前向输入
        x_b_in = x_b.clone().detach()#反向输入

        gcn_out_f = self.gnn_layers[0](x_f_in, batch_gated_edge_index,embedding=batch_node_embeddings)
        gcn_out_b = self.gnn_layers[1](x_b_in, batch_gated_edge_index_b,embedding=batch_node_embeddings_b)


        indexes = torch.arange(0,node_num).to(device)
        
        #前向预测
        f_out = gcn_out_f.contiguous()
        f_out = f_out.view(batch_num, node_num, -1)
        f_out = torch.mul(f_out, self.embedding(indexes))

        #反向预测
        b_out = gcn_out_b.contiguous()
        b_out = b_out.view(batch_num, node_num, -1)
        b_out = torch.mul(b_out, self.embedding_b(indexes))

        
        gdn_fusion = self.leaky_relu(self.gdn_fclayer(torch.concat((f_out,b_out), dim=-1))).permute(0,2,1)

        encoder_input = x_f.contiguous().reshape(batch_num, node_num, all_feature).permute(0,2,1)#(B*N,L)->(B,L,N)

        # 输入形状要是B,N,L,mask_f将观察值设置为0,取反后观察值为1
        inputs = {"X": encoder_input,
                "missing_mask": missing_mask}
        encoder_output,attn_weights = self.encoder_layers[0](inputs) #输出是B,L,N

        # the attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([missing_mask, attn_weights], dim=2))
        )  # namely term eta
        
        combat =  combining_weights*encoder_output  + (1 - combining_weights)*gdn_fusion
        encoder_imputed = combat*(1-missing_mask)  + X*missing_mask #用原始数据弥补非缺失数据
        encoder_imputed = encoder_imputed.contiguous().reshape(-1, node_num)#(B,L,N)->(B*L,N)


        gdn_fusion = gdn_fusion.contiguous().reshape(-1, node_num)#(B,L,N)->(B*L,N)
        encoder_output = encoder_output.contiguous().reshape(-1, node_num)#(B,L,N)->(B*L,N)
        combat = combat.contiguous().reshape(-1, node_num)#(B,L,N)->(B*L,N)
        missing_mask = missing_mask.reshape(batch_num*all_feature, -1)#掩码矩阵(B,L,N)->(B*L,N)
        indicating_mask = indicating_mask.reshape(batch_num*all_feature, -1)#掩码矩阵(B,L,N)->(B*L,N)

        return [gdn_fusion,encoder_output,combat,encoder_imputed],indicating_mask,missing_mask

