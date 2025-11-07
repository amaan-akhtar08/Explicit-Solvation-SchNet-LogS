
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def softplus(x):
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.clamp(x, min=0)

class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion on distances."""
    def __init__(self, n_rbf: int = 64, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, n_rbf)
        # Fix centers; learnable width (gamma)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0))  # width

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [n_edges]
        diff = d.unsqueeze(-1) - self.centers  # [n_edges, n_rbf]
        return torch.exp(-self.gamma * diff.pow(2))

class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, d):
        x = torch.clamp(d / self.cutoff, 0.0, 1.0)
        return 0.5 * (torch.cos(torch.pi * x) + 1.0) * (x < 1.0).float()

class CFConv(nn.Module):
    """Continuous-filter convolution as in SchNet."""
    def __init__(self, hidden_dim: int, n_rbf: int, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dense = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, edge_index, distances):
        # x: [n_atoms, hidden]
        # edge_index: [2, n_edges] (src->dst), with only intra-structure edges
        # distances: [n_edges]
        rbf = self.rbf(distances)  # [n_edges, n_rbf]
        W = self.filter_net(rbf)   # [n_edges, hidden]
        C = self.cutoff_fn(distances).unsqueeze(-1)  # [n_edges,1]
        W = W * C
        # message passing: m_j = sum_i ( W_ij * x_i )
        src, dst = edge_index[0], edge_index[1]
        m = self.dense(x[src]) * W  # [n_edges, hidden]
        # scatter-add to dst
        out = torch.zeros_like(x)
        out.index_add_(0, dst, m)
        return out

class SchNetBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_rbf: int, cutoff: float):
        super().__init__()
        self.cfconv = CFConv(hidden_dim, n_rbf, cutoff)
        self.act = nn.SiLU()
        self.dense = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, distances):
        v = self.cfconv(x, edge_index, distances)
        v = self.dense(self.act(v))
        return x + v  # residual

class SchNet(nn.Module):
    """
    Minimal SchNet with atom embedding, K interaction blocks, mean pooling, and
    a temperature-conditioned readout (FiLM).
    """
    def __init__(self, n_atom_types: int = 100, hidden_dim: int = 128, n_blocks: int = 5,
                 n_rbf: int = 64, cutoff: float = 5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim)
        self.blocks = nn.ModuleList([SchNetBlock(hidden_dim, n_rbf, cutoff) for _ in range(n_blocks)])
        # FiLM for temperature conditioning
        self.film = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2*hidden_dim)  # scale and bias
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.cutoff = cutoff

    @staticmethod
    def _neighbor_graph(pos, batch, cutoff: float):
        # Build radius graph naively (O(N^2) within each structure in the batch)
        # pos: [N,3], batch: [N] in {0..B-1}
        device = pos.device
        N = pos.size(0)
        # Compute pairwise within-batch with masking
        # We'll do this per structure to save memory
        edge_src = []
        edge_dst = []
        dists = []
        B = int(batch.max().item()) + 1 if N > 0 else 0
        for b in range(B):
            idx = torch.nonzero(batch == b, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            pb = pos[idx]  # [Nb,3]
            # pairwise distances
            diff = pb.unsqueeze(1) - pb.unsqueeze(0)  # [Nb,Nb,3]
            dist = torch.linalg.norm(diff, dim=-1)     # [Nb,Nb]
            mask = (dist > 0.0) & (dist <= cutoff)
            src_b, dst_b = torch.nonzero(mask, as_tuple=True)
            edge_src.append(idx[src_b])
            edge_dst.append(idx[dst_b])
            dists.append(dist[src_b, dst_b])
        if len(edge_src) == 0:
            edge_index = torch.zeros(2,0, dtype=torch.long, device=device)
            distances = torch.zeros(0, dtype=pos.dtype, device=device)
        else:
            edge_src = torch.cat(edge_src).to(device)
            edge_dst = torch.cat(edge_dst).to(device)
            distances = torch.cat(dists).to(device)
            edge_index = torch.stack([edge_src, edge_dst], dim=0)
        return edge_index, distances

    def forward(self, z, pos, batch, T):
        """
        Args:
          z: [N] atomic numbers (1..Z)
          pos: [N,3] positions (Å)
          batch: [N] sample index
          T: [B,1] Kelvin, one per structure in the batch
        """
        x = self.embedding(z)  # [N, hidden]
        edge_index, distances = self._neighbor_graph(pos, batch, cutoff=self.cutoff)
        for block in self.blocks:
            x = block(x, edge_index, distances)

        # mean pool per structure
        B = int(batch.max().item()) + 1 if z.numel() > 0 else T.size(0)
        pooled = torch.zeros(B, x.size(-1), device=x.device)
        pooled.index_add_(0, batch, x)
        # divide by counts to get mean
        counts = torch.bincount(batch, minlength=B).clamp(min=1).unsqueeze(-1)
        pooled = pooled / counts

        # Temperature conditioning via FiLM
        film_params = self.film(T)  # [B, 2H]
        gamma, beta = film_params.chunk(2, dim=-1)
        h = gamma * pooled + beta

        y = self.readout(h)  # [B,1], predicts LogS
        return y.squeeze(-1)

def batch_collate(samples: list) -> Dict[str, torch.Tensor]:
    """Collate a list of samples from ExplicitSolvDataset into a batch."""
    # concat atoms, keep batch vector
    z_list, pos_list, y_list, T_list, batch_list = [], [], [], [], []
    offset = 0
    for b_idx, s in enumerate(samples):
        z_list.append(s["z"])
        pos_list.append(s["pos"])
        y_list.append(s["y"])
        T_list.append(s["T"])
        batch_list.append(torch.full((s["z"].shape[0],), b_idx, dtype=torch.long))
    z = torch.cat(z_list, dim=0)
    pos = torch.cat(pos_list, dim=0)
    y = torch.cat(y_list, dim=0).view(-1,1)
    T = torch.stack(T_list, dim=0).view(-1,1)
    batch = torch.cat(batch_list, dim=0)
    return {"z": z, "pos": pos, "y": y, "T": T, "batch": batch}
