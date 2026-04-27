import torch
import torch.nn.functional as F

def build_adj_mask(edge_index, num_nodes, device):
    """
    Creates a sparse adjacency mask from edge_index.
    """
    row, col = edge_index
    val = torch.ones(edge_index.shape[1], device=device)
    adj = torch.sparse_coo_tensor(edge_index, val, (num_nodes, num_nodes))
    return adj


def spatial_smoothness_loss(z, edge_index, edge_weight=None):
    """
    Encourages nearby cells to have similar embeddings.
    L = sum_{i,j in edges} w_ij ||z_i - z_j||^2
    """
    row, col = edge_index
    if edge_weight is None:
        diff = z[row] - z[col]
        return (diff**2).sum(dim=1).mean()
    else:
        diff = z[row] - z[col]
        return (edge_weight.view(-1, 1) * (diff**2)).sum(dim=1).mean()
