from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

def build_spatial_graph(coords: np.ndarray, k: int = 6):
    """
    Build a k-NN graph from spatial coordinates.
    Returns edge_index [2, E] and edge_weight [E] (all ones for now).
    """
    log.info(f"Building spatial k-NN graph (k={k}, N={len(coords)}) …")
    tree = KDTree(coords)
    dist, idx = tree.query(coords, k=k+1)
    
    # Exclude self-loops (index 0 is usually self)
    row = np.repeat(np.arange(len(coords)), k)
    col = idx[:, 1:].flatten()
    
    edge_index = np.stack([row, col], axis=0)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    
    return edge_index, edge_weight


def build_spatial_features(
    coords: np.ndarray, 
    cell_types: np.ndarray,
    ks: tuple[int, ...] = (10, 30, 60)
):
    """
    Computes neighborhood cell-type composition at multiple scales (ks).
    """
    N = len(coords)
    tree = KDTree(coords)
    
    # Composition features
    log.info(f"Computing neighborhood composition at scales {ks} …")
    unique_cts = sorted(set(cell_types))
    ct_map = {ct: i for i, ct in enumerate(unique_cts)}
    ct_ints = np.array([ct_map[ct] for ct in cell_types])
    
    comp_feats = []
    for k in ks:
        _, idx = tree.query(coords, k=k)
        # For each cell, count CTs in its k-neighbors
        counts = np.zeros((N, len(unique_cts)))
        for i in range(N):
            neigh_cts = ct_ints[idx[i]]
            counts[i] = np.bincount(neigh_cts, minlength=len(unique_cts))
        comp_feats.append(counts / k)
    
    spat_X = np.concatenate(comp_feats, axis=1).astype(np.float32)
    
    return spat_X
