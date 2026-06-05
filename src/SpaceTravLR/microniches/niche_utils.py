"""
Utility functions for identifying functional microniches using the SpatialFunctionalModel.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .functional_niches.dataset import build_spatial_graph, build_spatial_features
from .functional_niches.functional_model import train_functional
from .functional_niches.cluster import cluster_embeddings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Data Loading Utilities ──────────────────────────────────────────

def _get_schema(p: Path) -> list[str]:
    suffix = p.suffix.lower()
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq
            return pq.read_schema(p).names
        else:
            # Try loading as Arrow/Feather
            with pa.memory_map(str(p), "r") as src:
                import pyarrow.ipc as ipc
                try:
                    return ipc.open_file(src).schema.names
                except Exception:
                    # If ipc failed, it might be a parquet file mislabeled as feather
                    import pyarrow.parquet as pq
                    return pq.read_schema(p).names
    except Exception as e:
        log.error(f"Failed to read schema for {p}: {e}")
        return []


def _load_one(path: Path, cell_ids: list[str], mod_vocab: dict[str, int]):
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            try:
                tbl = feather.read_table(path)
                df = tbl.to_pandas()
            except Exception:
                # Fallback to parquet if feather fails
                df = pd.read_parquet(path)
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        return None
    
    # Identify the ID column
    # Priority: 1. 'CellID' column, 2. Unique named index, 3. Unique unnamed index (if not simple range), 4. First column
    if "CellID" in df.columns:
        id_col = "CellID"
    elif df.index.is_unique and (df.index.name is not None or not isinstance(df.index, pd.RangeIndex)):
        # If index is unique and named OR unique and not just 0,1,2..., use it
        df = df.reset_index()
        id_col = df.columns[0]
    else:
        # Fallback to the first column
        id_col = df.columns[0]
        log.warning(f"Using '{id_col}' as the ID column for {path.name}. If this is wrong, rename your ID column to 'CellID'.")

    beta_cols = [c for c in df.columns if c.startswith("beta_")]
    if not beta_cols:
        return None
        
    # Handle duplicate IDs if they still exist after identifying the ID column
    if df[id_col].duplicated().any():
        log.warning(f"Duplicate IDs found in column '{id_col}' of {path.name}. Averaging values.")
        df = df.groupby(id_col)[beta_cols].mean().reset_index()

    df = df.set_index(id_col).reindex(cell_ids)
    betas = df[beta_cols].fillna(0).values.astype(np.float32)
    mod_idx = np.array([mod_vocab[c] for c in beta_cols], dtype=np.int64)
    gene_name = path.stem.replace("_betadata", "")
    return gene_name, mod_idx, betas


def build_vocab_parallel(paths, n_workers=16):
    """Builds a vocabulary of all modulator names across multiple Feather or Parquet files."""
    all_mods: set[str] = set()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for cols in ex.map(_get_schema, paths):
            all_mods.update(c for c in cols if c.startswith("beta_"))
    return {name: i for i, name in enumerate(sorted(all_mods))}


def load_regulatory_data(data_paths, cell_ids, n_workers=16):
    """
    Parallel loading of gene-specific beta values from Feather or Parquet files.
    Returns gene_activity[N, G] and gene_names[G].
    """
    mod_vocab = build_vocab_parallel(data_paths, n_workers=n_workers)
    
    gene_list: list[np.ndarray] = []
    gene_names: list[str]       = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_load_one, p, cell_ids, mod_vocab): p for p in data_paths}
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            gene_name, mod_idx, betas = res
            # Mean |beta| per cell per gene (average across all modulators for that gene)
            gene_list.append(np.abs(betas).mean(axis=1))   
            gene_names.append(gene_name)
            
    gene_activity = np.stack(gene_list, axis=1).astype(np.float32)   # [N, G]
    return gene_activity, gene_names


# ── Pipeline Wrapper ───────────────────────────────────────────────

def identify_microniches(
    adata,
    feather_dir: str | Path,
    target_genes: set[str] | list[str],
    output_dir: str | Path,
    cell_type_col: str = "cell_type",
    ref_annot_col: str = "ref_niche",
    n_workers: int = 16,
    train_params: dict | None = None,
    resolutions: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30),
    device: str = "auto"
):
    """
    Full pipeline to identify functional microniches.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial coordinates in .obsm['spatial'].
    feather_dir : str or Path
        Directory containing gene_betadata.feather files.
    target_genes : set or list
        List of genes to include in the functional model.
    output_dir : str or Path
        Directory to save results and plots.
    cell_type_col : str, default 'cell_type'
        Column in adata.obs with cell type labels.
    ref_annot_col : str, default 'ref_niche'
        Reference column for evaluation (optional).
    train_params : dict, optional
        Override default training parameters.
    resolutions : tuple, default (0.15, 0.2, 0.25, 0.3)
        Leiden resolutions to evaluate.
    device : str, default 'auto'
        'cpu', 'cuda', or 'auto'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_ids       = list(adata.obs_names)
    spatial_coords = adata.obsm["spatial"].astype(np.float32)
    cell_type      = adata.obs[cell_type_col].values.astype(str)
    
    # 1. Load target data (detect feather or parquet, case-insensitive)
    feather_files = list(Path(feather_dir).glob("*_betadata.[fF][eE][aA][tT][hH][eE][rR]"))
    parquet_files = list(Path(feather_dir).glob("*_betadata.[pP][aA][rR][qQ][uU][eE][tT]"))
    all_files = sorted(feather_files + parquet_files)

    
    data_paths = [p for p in all_files
                  if p.stem.replace("_betadata","") in target_genes]
    
    if not data_paths:
        raise ValueError(f"No data files (.feather or .parquet) found in {feather_dir} matching target genes.")

    log.info(f"Loading {len(data_paths)} files ...")
    gene_activity, gene_names = load_regulatory_data(data_paths, cell_ids, n_workers=n_workers)

    # 2. Spatial features and graph
    log.info("Building spatial features and graph …")
    spat_X = build_spatial_features(
        spatial_coords, cell_type,
        ks=(10, 30, 60)
    )
    edge_index, edge_weight = build_spatial_graph(spatial_coords, k=6)

    # 3. Training
    log.info("Training SpatialFunctionalModel …")
    
    # Default training parameters
    params = {
        "hidden_dim": 64,
        "mlp_layers": 2,
        "gcn_layers": 2,
        "epochs": 800,
        "lr": 1e-3,
        "w_triplet": 1.0,
        "w_rec": 0.05,
        "w_smooth": 0.3,
        "w_nbr_comp": 0.5,
        "log_every": 100,
    }
    if train_params:
        params.update(train_params)

    z = train_functional(
        beta_X      = gene_activity,
        spat_X      = spat_X,
        rec_target  = gene_activity,
        edge_index  = edge_index,
        edge_weight = edge_weight,
        cell_ids    = cell_ids,
        output_dir  = str(output_dir),
        device_str  = device,
        **params
    )

    # 4. Evaluation and Clustering
    log.info("Clustering and Evaluation …")
    true_labels = None
    if ref_annot_col in adata.obs.columns:
        true_labels = adata.obs[ref_annot_col].values.astype(str)
        best = evaluate_niches(z, true_labels, resolutions=resolutions)
        log.info(f"Best result: ARI={best['ari']:.4f}  NMI={best['nmi']:.4f}  n={best['n']}")
    else:
        # Just pick a middle resolution if no ground truth
        res = resolutions[len(resolutions)//2]
        r = cluster_embeddings(z, resolutions=[res])
        best = {"labels": r[res], "res": res, "n": len(set(r[res])), "ari": 0, "nmi": 0}

    niche_labels = best["labels"]
    
    # 5. Save results to AnnData
    adata.obs["functional_niche"] = niche_labels
    adata.obsm["X_functional_niche"] = z
    
    # Save metadata to disk
    pd.DataFrame({
        "CellID": cell_ids, 
        "niche": niche_labels
    }).to_parquet(output_dir / "niche_labels.parquet", index=False)
    np.save(str(output_dir / "embeddings.npy"), z)

    return z, niche_labels


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_niches(z, true_labels, resolutions=(0.15, 0.20, 0.25, 0.30)):
    """Evaluate clustering performance against ground truth labels."""
    best = {"ari": -1.0, "nmi": 0.0, "res": None, "n": 0, "labels": None}
    for res in resolutions:
        r    = cluster_embeddings(z, resolutions=[res])
        pred = r[res].astype(str)
        ari  = adjusted_rand_score(true_labels, pred)
        nmi  = normalized_mutual_info_score(true_labels, pred, average_method="arithmetic")
        if ari > best["ari"]:
            best = {"ari": ari, "nmi": nmi, "res": res,
                    "n": len(set(pred)), "labels": pred}
    return best


# ── Plotting ──────────────────────────────────────────────────────

_PAL = (["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4",
          "#f032e6","#bfef45","#469990","#dcbeff","#9a6324","#800000",
          "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#ffe119",
          "#4e9ddb","#c0a040"])


def _pal(labels):
    unique = sorted(set(labels), key=lambda x: (int(x) if x.isdigit() else 999, x))
    cm = {k: _PAL[i % len(_PAL)] for i, k in enumerate(unique)}
    return cm, [cm[l] for l in labels]


def _handles(cm):
    return [plt.Line2D([0],[0], marker="o", color="w",
                        markerfacecolor=v, markersize=7, label=str(k))
            for k, v in cm.items()]


def visualize_niches(
    adata,
    niche_labels,
    z,
    output_dir: str | Path,
    tfh_dist: np.ndarray | None = None,
    cell_type_col: str = "cell_type",
    ref_annot_col: str | None = None,
    compute_umap: bool = True
):
    """Generate spatial and UMAP visualizations of the niches."""
    output_dir = Path(output_dir)
    spatial_coords = adata.obsm["spatial"]
    cell_type      = adata.obs[cell_type_col].values.astype(str)
    
    niche_cm, niche_col = _pal(niche_labels)
    ct_cm,    ct_col    = _pal(cell_type)
    
    # ── 1. Spatial Plot ──────────────────────────────────────────
    n_cols = 2 + (1 if ref_annot_col else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(10 * n_cols, 9))
    
    # Niches
    axes[0].scatter(spatial_coords[:,0], spatial_coords[:,1],
                   c=niche_col, s=5, alpha=0.85, rasterized=True)
    axes[0].legend(handles=_handles(niche_cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, frameon=True, ncol=max(1, len(niche_cm)//15))
    axes[0].set_title(f"Functional microniches ({len(niche_cm)})", fontsize=11)
    
    # Cell Type
    axes[1].scatter(spatial_coords[:,0], spatial_coords[:,1],
                   c=ct_col, s=5, alpha=0.85, rasterized=True)
    axes[1].legend(handles=_handles(ct_cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, frameon=True, ncol=max(1, len(ct_cm)//15))
    axes[1].set_title("cell_type", fontsize=11)
    
    # Reference if available
    if ref_annot_col:
        ct2_cm, ct2_col = _pal(adata.obs[ref_annot_col].astype(str))
        axes[2].scatter(spatial_coords[:,0], spatial_coords[:,1],
                       c=ct2_col, s=5, alpha=0.85, rasterized=True)
        axes[2].legend(handles=_handles(ct2_cm), bbox_to_anchor=(1.01,1), loc="upper left",
                      fontsize=7, frameon=True, ncol=max(1, len(ct2_cm)//15))
        axes[2].set_title(f"{ref_annot_col} (reference)", fontsize=11)

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal"); ax.invert_yaxis()
        
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_niches.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ── 2. UMAP ──────────────────────────────────────────────────
    if compute_umap:
        log.info("Computing UMAP of embeddings …")
        import umap as umap_lib
        coords = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z)
        np.save(output_dir / "umap_coords.npy", coords)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        axes[0].scatter(coords[:,0], coords[:,1], c=niche_col, s=4, alpha=0.7, rasterized=True)
        axes[0].set_title("Niches in Embedding Space", fontsize=11)
        
        axes[1].scatter(coords[:,0], coords[:,1], c=ct_col, s=4, alpha=0.7, rasterized=True)
        axes[1].set_title("Cell Types in Embedding Space", fontsize=11)
        
        for ax in axes:
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            
        plt.tight_layout()
        plt.savefig(output_dir / "umap_niches.png", dpi=180, bbox_inches="tight")
        plt.close()

    log.info(f"Visualizations saved to {output_dir}")
