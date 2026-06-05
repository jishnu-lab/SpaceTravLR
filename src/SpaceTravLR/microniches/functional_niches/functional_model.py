import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

class SpatialFunctionalModel(nn.Module):
    def __init__(self, beta_dim, spat_dim, hidden_dim=64, z_dim=32, n_layers=2):
        super().__init__()
        # Encoder for individual cell beta features
        layers = [nn.Linear(beta_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.beta_enc = nn.Sequential(*layers)
        
        # Encoder for spatial context
        spat_layers = [nn.Linear(spat_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            spat_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.spat_enc = nn.Sequential(*spat_layers)
        
        # Combined embedding
        self.z_head = nn.Linear(hidden_dim * 2, z_dim)
        
        # Decoder for reconstruction
        self.decoder = nn.Linear(z_dim, beta_dim)

    def forward(self, beta_X, spat_X):
        b_h = self.beta_enc(beta_X)
        s_h = self.spat_enc(spat_X)
        h = torch.cat([b_h, s_h], dim=1)
        z = self.z_head(h)
        rec = self.decoder(z)
        return z, rec


def spatial_triplet_loss(z, edge_index, margin=1.0):
    # Very simple version: sample some random negatives
    row, col = edge_index
    pos_dist = torch.norm(z[row] - z[col], p=2, dim=1)
    
    # Random negatives
    neg_idx = torch.randint(0, z.shape[0], (row.shape[0],), device=z.device)
    neg_dist = torch.norm(z[row] - z[neg_idx], p=2, dim=1)
    
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

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


def train_functional(
    beta_X, spat_X, rec_target,
    edge_index, edge_weight, cell_ids, output_dir,
    hidden_dim=64, mlp_layers=2, gcn_layers=2,
    epochs=800, lr=1e-3, w_triplet=1.0, w_rec=0.05,
    w_smooth=0.3, w_nbr_comp=0.5,
    device_str="auto", log_every=100, **kwargs
):
    device = torch.device("cuda" if torch.cuda.is_available() and device_str=="auto" else "cpu")
    if device_str != "auto":
        device = torch.device(device_str)
        
    N, G = beta_X.shape
    S = spat_X.shape[1]
    
    model = SpatialFunctionalModel(G, S, hidden_dim=hidden_dim, z_dim=hidden_dim//2, n_layers=mlp_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    b_X = torch.from_numpy(beta_X).to(device)
    s_X = torch.from_numpy(spat_X).to(device)
    r_T = torch.from_numpy(rec_target).to(device)
    e_I = torch.from_numpy(edge_index).to(device)
        
    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        z, rec = model(b_X, s_X)
        
        loss_rec = F.mse_loss(rec, r_T)
        loss_triplet = spatial_triplet_loss(z, e_I)
        loss_smooth = spatial_smoothness_loss(z, e_I)
        
        total_loss = w_rec * loss_rec + w_triplet * loss_triplet + w_smooth * loss_smooth
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % log_every == 0:
            log.info(f"Epoch {epoch}: loss={total_loss.item():.4f} (rec={loss_rec.item():.4f}, tri={loss_triplet.item():.4f})")
            
    model.eval()
    with torch.no_grad():
        z, _ = model(b_X, s_X)
        return z.cpu().numpy()
