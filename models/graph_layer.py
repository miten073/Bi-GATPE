import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU , LayerNorm
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import time
import math

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs) # 调用父类MessagePassing的构造函数

        self.in_channels = in_channels # 输入特征的通道数
        self.out_channels = out_channels # 输出特征的通道数

        self.heads = heads # 注意力头的数量
        self.concat = concat # 指定是否将头的结果拼接在一起
        self.negative_slope = negative_slope # LeakyReLU激活函数的负斜率
        self.dropout = dropout # Dropout概率

        # self.num = 0

        self.__alpha__ = None # 存储注意力权重的变量


        self.lin = Linear(in_channels, heads * out_channels, bias=False) # 线性变换层
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels)) # 注意力系数att_i
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels)) # 注意力系数att_j
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels)) # 嵌入的注意力系数att_em_i
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels)) # 嵌入的注意力系数att_em_j

        if bias and concat: # 如果设置了偏置项并且拼接结果
            self.bias = Parameter(torch.Tensor(heads * out_channels)) # 偏置项
        elif bias and not concat: # 如果设置了偏置项但不拼接结果
            self.bias = Parameter(torch.Tensor(out_channels)) # 偏置项
        else: # 如果没有设置偏置项
            self.register_parameter('bias', None) # 注册一个空的参数

        self.reset_parameters() # 初始化模型参数

        self.layer_norm = LayerNorm(out_channels) 

    def reset_parameters(self) :

        glorot(self.att_i) # 使用Glorot方法初始化att_i的权重
        glorot(self.att_j) # 使用Glorot方法初始化att_j的权重
        zeros(self.att_em_i) # 将att_em_i的权重初始化为零
        zeros(self.att_em_j) # 将att_em_j的权重初始化为零
        zeros(self.bias) # 将偏置项初始化为零

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        if torch.is_tensor(x): # 如果输入是张量
            x = self.lin(x) # 进行线性变换
            x = (x, x) # 复制为元组
        else: # 如果输入是元组
            x = (self.lin(x[0]), self.lin(x[1])) # 对两个元素进行线性变换
        edge_index, _ = remove_self_loops(edge_index) # 移除自环边
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim)) # 添加自环边
        
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights) # 进行消息传递message



        #(B*N,H,E)

        if self.concat: # 如果需要拼接结果
            out = out.view(-1, self.heads * self.out_channels) # 将结果展平为二维张量
        else: # 如果需要进行均值池化
            out = out.mean(dim=1) # 对结果进行均值池化

        #残差连接
        # out += x[0]
        # out = self.layer_norm(out) 
        
        #(B*N,H*E)

        if self.bias is not None: # 如果设置了偏置项
            out = out + self.bias # 加上偏置项

        #(B*N,H*E)

        if return_attention_weights: # 如果需要返回注意力权重
            alpha, self.__alpha__ = self.__alpha__, None # 保存注意力权重并清空
            return out, (edge_index, alpha) # 返回结果和注意力权重
        else: # 如果不需要返回注意力权重
            return out # 返回结果

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels) # 将节点特征重塑为三维张量
        x_j = x_j.view(-1, self.heads, self.out_channels) # 将节点特征重塑为三维张量

        if embedding is not None: # 如果存在嵌入特征
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]] # 获取嵌入特征 embedding(224,64)
            embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1) # 将嵌入特征复制为与头数相同
            embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1) # 将嵌入特征复制为与头数相同
            key_i = torch.cat((x_i, embedding_i), dim=-1) # 将节点特征和嵌入特征拼接在一起 Xi，Vi 
            key_j = torch.cat((x_j, embedding_j), dim=-1) # 将节点特征和嵌入特征拼接在一起 Whi

        d_k = key_i.shape[-1]
        
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1) # 将att_i和att_em_i拼接在一起
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1) # 将att_j和att_em_j拼接在一起

        #GATv1
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1) # 计算注意力系数alpha sum(-1)逐行求和
        alpha = alpha.view(-1, self.heads, 1) # 将alpha的形状重塑为三维张量
        alpha = F.leaky_relu(alpha, self.negative_slope) # 使用LeakyReLU激活函数

        # # GATv2
        # alpha = F.leaky_relu((key_i + key_j), self.negative_slope)
        # dist1 = torch.cdist(key_i, key_j) 
        # alpha = (alpha * cat_att_i).sum(-1) + (alpha * cat_att_j).sum(-1) + 1/(dist1+1e-6).sum(-1) #计算注意力是加入了节点的距离，精度基本不变，收敛快了
        # # alpha = (alpha * cat_att_i).sum(-1) + (alpha * cat_att_j).sum(-1)  #计算注意力是加入了节点的距离，精度基本不变，收敛快了

        # #1.5.0torch版本
        # alpha = softmax(alpha, edge_index_i, size_i) 
        #其他torch版本
        self.node_dim=0
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights: # 如果需要返回注意力权重
            self.__alpha__ = alpha # 保存注意力权重

        alpha = F.dropout(alpha, p=self.dropout, training=self.training) # 使用dropout进行正则化
        # m =  x_j * alpha.view(-1, self.heads, 1)
        m =  (x_j * alpha.view(-1, self.heads, 1))
        return m # 返回经过注意力系数加权的节点特征

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
