import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
import warnings

Tensor = torch.Tensor

def adjacency_to_edge_index(adjacency_matrix):
    edge_index = []
    num_nodes = adjacency_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # 只考虑上三角部分
            if adjacency_matrix[i, j] != 0:
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



def scaled_laplacian(self, node_embeddings, is_eval=False):
    # Normalized graph Laplacian function.
    # :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    # :return: np.matrix, [n_route, n_route].
    # learned graph
    node_num = self.num_node
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
    mask = torch.eye(node_num, node_num).bool().cuda()
    adj.masked_fill_(mask, 0)  #根据给定的掩码 mask，将 adj 张量中与掩码对应位置的元素设置为 0。
    
    return adjacency_to_edge_index(adj)

