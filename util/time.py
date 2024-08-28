import time
import math
from datetime import datetime
from pytz import utc, timezone
from util.env import get_device, set_device

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSincePlus(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timestamp2str(sec, fmt, tz):
    return datetime.fromtimestamp(sec).astimezone(tz).strftime(fmt)


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
import warnings

Tensor = torch.Tensor

def adjacency_to_edge_index(adj):
    edge_index = []
    num_nodes = adj.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] != 0:
                edge_index.append([i, j])
    return torch.tensor(edge_index).t().contiguous()

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret



def scaled_laplacian(num_node, node_embeddings, is_eval=False):
    # Normalized graph Laplacian function.
    # :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    # :return: np.matrix, [n_route, n_route].
    # learned graph
    node_num = num_node
    learned_graph = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
    norm = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)
    norm = torch.mm(norm, norm.transpose(0, 1))
    learned_graph = learned_graph / norm
    
    learned_graph = (learned_graph + 1) / 2.
    # learned_graph = F.sigmoid(learned_graph)
    learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
    
    # make the adj sparse
    if is_eval:
        adj = gumbel_softmax(learned_graph, tau=1, hard=True)
    else:
        adj = gumbel_softmax(learned_graph, tau=1, hard=True)
    adj = adj[:, :, 0].clone().reshape(node_num, -1)
    # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
    device = get_device()
    mask = torch.eye(node_num, node_num).bool().to(device)
    adj.masked_fill_(mask, 0)  #根据给定的掩码 mask，将 adj 张量中与掩码对应位置的元素设置为 0。
    
    return adjacency_to_edge_index(adj)

# if __name__ == "__main__":
#     # 假设 node_embeddings 是一个包含节点嵌入的张量，示例数据如下：
#     node_embeddings = torch.tensor([[0.1, 0.1], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])

    
#     # 调用 scaled_laplacian 函数
#     adjacency_matrix = scaled_laplacian(num_node=4, node_embeddings=node_embeddings, is_eval=False)

#     print(adjacency_matrix)

