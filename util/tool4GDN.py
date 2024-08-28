import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



def get_edge_index(data, k):
    """
    为单个批次构建图的边索引
    输入:
        data: (N, L) 张量,表示该批次中 N 个变量,每个变量长度为 L
        k: 整数,指定 KNN 的 k 值
    输出:
        edge_index: (2, E) 张量,表示该批次图的边索引
    """
    N, L = data.shape
    
    # 计算距离矩阵
    dist_matrix = squareform(pdist(data))
    dist_matrix = torch.from_numpy(dist_matrix)
    
    # 应用 KNN
    knn_matrix = dist_matrix.topk(k+1, largest=False, dim=1)[1][:, 1:] # 排除自身
    adj_matrix = torch.zeros(N, N)
    adj_matrix = adj_matrix.scatter(1, knn_matrix, 1) # 构建邻接矩阵
    
    # 构建 edge_index
    edge_index = torch.nonzero(adj_matrix).t()
    
    return edge_index

def get_knn_batch_edge_index(data, k):
    """
    为多个批次构建图的边索引,并进行节点偏移
    输入:
        data: (B, N, L) 张量,表示 B 个批次,每个批次有 N 个变量,每个变量长度为 L
        k: 整数,指定 KNN 的 k 值
    输出:
        batch_edge_index: (2, E) 张量,表示多个批次图的边索引,节点已进行偏移
    """
    B, N, L = data.shape
    batch_edge_indices = []
    
    for i in range(B):
        # 构建单个批次的边索引
        edge_index = get_edge_index(data[i], k)
        
        # 对节点进行偏移
        edge_index += i*N
        
        batch_edge_indices.append(edge_index)
    
    batch_edge_index = torch.cat(batch_edge_indices, dim=1)
    
    return batch_edge_index

def compute_edge_index(data, topk):
    """
    计算余弦相似度，并获取边索引
    输入:
        data: (node_num, dim) 张量，表示节点特征矩阵
        topk: 整数，指定每个节点的邻居数量
    输出:
        edge_index: (2, E) 张量，表示边索引
    """
    device = data.device

    node_num, dim = data.shape

    weights_arr = data.detach().clone() # 节点特征矩阵

    cos_ji_mat = torch.matmul(weights_arr, weights_arr.T) # 计算余弦相似度矩阵
    normed_mat = torch.matmul(weights_arr.norm(dim=-1).view(-1,1), weights_arr.norm(dim=-1).view(1,-1))
    cos_ji_mat = cos_ji_mat / normed_mat # 归一化

    topk_indices_ji = torch.topk(cos_ji_mat, topk, dim=-1)[1] # 获取每行的前topk个最大值的索引

    gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk).flatten().to(device).unsqueeze(0)
    gated_j = topk_indices_ji.flatten().unsqueeze(0)
    edge_index = torch.cat((gated_j, gated_i), dim=0)

    return edge_index

class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        attn_dropout=0.1,
        **kwargs
    ):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = kwargs["diagonal_attention_mask"]
        self.device = kwargs["device"]
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=mask_time
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()

# class BidirectionalScaledDotProductAttention(nn.Module):
#     """Scaled dot-product attention with support for bidirectional attention."""

#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)

#     def forward(self, q, k, v, attn_mask=None):
#         attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
#         if attn_mask is not None:
#             attn = attn.masked_fill(attn_mask == 1, -1e9)
#         attn = self.dropout(F.softmax(attn, dim=-1))
#         output = torch.matmul(attn, v)
#         return output, attn

# class BidirectionalMultiHeadAttention(nn.Module):
#     """Multi-head attention with support for bidirectional attention."""

#     def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
#         super().__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

#         self.attention = BidirectionalScaledDotProductAttention(d_k**0.5, attn_dropout)
#         self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

#     def forward(self, q, k, v, attn_mask=None):
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

#         if attn_mask is not None:
#             attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)

#         v, attn_weights = self.attention(q, k, v, attn_mask)

#         v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#         v = self.fc(v)
#         return v, attn_weights

# class BidirectionalEncoderLayer(nn.Module):
#     def __init__(self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, **kwargs):
#         super(BidirectionalEncoderLayer, self).__init__()

#         self.diagonal_attention_mask = kwargs["diagonal_attention_mask"]
#         self.device = kwargs["device"]
#         self.d_time = d_time
#         self.d_feature = d_feature

#         self.layer_norm = nn.LayerNorm(d_model)
#         self.slf_attn = BidirectionalMultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
#         self.dropout = nn.Dropout(dropout)
#         self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

#     def forward(self, enc_input):
#         residual = enc_input
#         enc_input = self.layer_norm(enc_input)

#         if self.diagonal_attention_mask:
#             attn_mask = torch.ones((enc_input.size(0), 1, 1, self.d_time), device=self.device)
#             attn_mask = torch.triu(attn_mask, diagonal=1)
#         else:
#             attn_mask = None

#         enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask)
#         enc_output = self.dropout(enc_output)
#         enc_output += residual

#         enc_output = self.pos_ffn(enc_output)
#         return enc_output, attn_weights

# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, d_in, d_hid, dropout=0.1):
#         super().__init__()
#         self.w_1 = nn.Linear(d_in, d_hid)
#         self.w_2 = nn.Linear(d_hid, d_in)
#         self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         residual = x
#         x = self.layer_norm(x)
#         x = self.w_2(F.relu(self.w_1(x)))
#         x = self.dropout(x)
#         x += residual
#         return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_hid, n_position=200):
#         super(PositionalEncoding, self).__init__()
#         self.register_buffer(
#             "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
#         )

#     def _get_sinusoid_encoding_table(self, n_position, d_hid):
#         def get_position_angle_vec(position):
#             return [
#                 position / np.power(10000, 2 * (hid_j // 2) / d_hid)
#                 for hid_j in range(d_hid)
#             ]

#         sinusoid_table = np.array(
#             [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
#         )
#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)

#     def forward(self, x):
#         return x + self.pos_table[:, : x.size(1)].clone().detach()

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TemporalConvLayer(in_channels if i == 0 else out_channels, out_channels, kernel_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x