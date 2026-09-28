"""Message-passing GNN regressor on neuron graphs (plain PyTorch; torch_geometric optional).

The model is a GraphSAGE-style encoder (mean aggregation over neighbours, residual MLP updates)
followed by global mean + max pooling and an MLP head that predicts all ephys targets jointly.
``torch`` is imported lazily so the package imports without it.

Typical use::

    from morph2ephys.gnn import GNNRegressor, train_regressor, predict
    model, history = train_regressor(train_graphs, Y_train, val_graphs, Y_val, n_targets=Y_train.shape[1])
    Y_hat = predict(model, test_graphs)

For transfer, call ``train_regressor(..., init_model=model_mouse, freeze_encoder=True)`` to fine-tune
only the head on human cells.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .swc_graph import NeuronGraph

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    nn = object  # type: ignore
    HAVE_TORCH = False


def collate(graphs: Sequence[NeuronGraph]):
    """Disjoint union of graphs -> (x, edge_index, batch) tensors."""
    if not HAVE_TORCH:
        raise ImportError("torch is required for the GNN")
    xs, eis, batch = [], [], []
    offset = 0
    for k, g in enumerate(graphs):
        xs.append(torch.as_tensor(g.x, dtype=torch.float32))
        eis.append(torch.as_tensor(g.edge_index, dtype=torch.long) + offset)
        batch.append(torch.full((g.n_nodes,), k, dtype=torch.long))
        offset += g.n_nodes
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(batch)


if HAVE_TORCH:

    class SAGELayer(nn.Module):
        """h_i' = ReLU(W_self h_i + W_nbr mean_{j in N(i)} h_j) with a residual connection."""

        def __init__(self, dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.lin_self = nn.Linear(dim, dim)
            self.lin_nbr = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.dropout = dropout

        def forward(self, h: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            src, dst = edge_index
            agg = torch.zeros_like(h).index_add_(0, dst, h[src])
            deg = torch.zeros(h.shape[0], device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            agg = agg / deg.clamp(min=1).unsqueeze(1)
            out = F.relu(self.lin_self(h) + self.lin_nbr(agg))
            out = F.dropout(out, self.dropout, self.training)
            return self.norm(h + out)

    class GNNRegressor(nn.Module):
        """Encoder (input MLP + L SAGE layers) -> mean||max pooling -> MLP head."""

        def __init__(self, in_dim: int, n_targets: int, hidden: int = 128, n_layers: int = 4,
                     dropout: float = 0.1) -> None:
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
            self.layers = nn.ModuleList([SAGELayer(hidden, dropout) for _ in range(n_layers)])
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, n_targets))

        def encode(self, x, edge_index, batch) -> "torch.Tensor":
            h = self.inp(x)
            for layer in self.layers:
                h = layer(h, edge_index)
            n_graphs = int(batch.max().item()) + 1
            mean = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, batch, h)
            counts = torch.zeros(n_graphs, device=h.device).index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
            mean = mean / counts.clamp(min=1).unsqueeze(1)
            mx = torch.full((n_graphs, h.shape[1]), -1e9, device=h.device).scatter_reduce(
                0, batch.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=True)
            return torch.cat([mean, mx], dim=1)

        def forward(self, x, edge_index, batch) -> "torch.Tensor":
            return self.head(self.encode(x, edge_index, batch))

        def node_attributions(self, graph: NeuronGraph, target_idx: int) -> np.ndarray:
            """Gradient x input attribution of one target w.r.t. node features (summed over features)."""
            self.eval()
            x, ei, b = collate([graph])
            x = x.requires_grad_(True)
            out = self.forward(x, ei, b)[0, target_idx]
            out.backward()
            return (x.grad * x).sum(dim=1).detach().cpu().numpy()


def _batches(n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for k in range(0, n, batch_size):
        yield idx[k:k + batch_size]


def train_regressor(train_graphs: Sequence[NeuronGraph], Y_train: np.ndarray,
                    val_graphs: Optional[Sequence[NeuronGraph]] = None, Y_val: Optional[np.ndarray] = None,
                    n_targets: Optional[int] = None, hidden: int = 128, n_layers: int = 4, epochs: int = 200,
                    batch_size: int = 16, lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 25,
                    init_model=None, freeze_encoder: bool = False, seed: int = 0,
                    device: Optional[str] = None) -> Tuple["GNNRegressor", Dict[str, List[float]]]:
    """Train with Huber loss on z-scored targets (NaN targets are masked); early stopping on validation."""
    if not HAVE_TORCH:
        raise ImportError("torch is required for the GNN")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Y_train = np.asarray(Y_train, np.float32)
    n_targets = n_targets or Y_train.shape[1]
    model = init_model if init_model is not None else GNNRegressor(train_graphs[0].x.shape[1], n_targets, hidden, n_layers)
    model = model.to(device)
    if freeze_encoder:
        for p in list(model.inp.parameters()) + list(model.layers.parameters()):
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    Yt = torch.as_tensor(Y_train, device=device)
    history: Dict[str, List[float]] = {"train": [], "val": []}
    best_state, best_val, bad = None, np.inf, 0
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for idx in _batches(len(train_graphs), batch_size, rng):
            x, ei, b = collate([train_graphs[i] for i in idx])
            x, ei, b = x.to(device), ei.to(device), b.to(device)
            pred = model(x, ei, b)
            y = Yt[idx]
            mask = torch.isfinite(y)
            loss = F.huber_loss(pred[mask], y[mask], delta=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss.item()) * len(idx)
        history["train"].append(tot / len(train_graphs))
        if val_graphs is not None and Y_val is not None:
            val_pred = predict(model, val_graphs, device=device)
            yv = np.asarray(Y_val, np.float32)
            m = np.isfinite(yv)
            val_loss = float(np.mean((val_pred[m] - yv[m]) ** 2))
            history["val"].append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val, bad = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model, graphs: Sequence[NeuronGraph], batch_size: int = 32, device: Optional[str] = None) -> np.ndarray:
    if not HAVE_TORCH:
        raise ImportError("torch is required for the GNN")
    device = device or next(model.parameters()).device
    model.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, len(graphs), batch_size):
            x, ei, b = collate(graphs[k:k + batch_size])
            outs.append(model(x.to(device), ei.to(device), b.to(device)).cpu().numpy())
    return np.vstack(outs) if outs else np.empty((0, 0))


__all__ = ["HAVE_TORCH", "collate", "train_regressor", "predict"] + (["GNNRegressor", "SAGELayer"] if HAVE_TORCH else [])
