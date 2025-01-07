import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
import umap

from .layout import plot_quiver, plot_vectorfield
from .shift import *


def estimate_transitions_2D(adata, delta_X, embedding, layout_embedding, annot=None, normalize=True, 
n_neighbors=200, grid_scale=1, vector_scale=1, n_jobs=1, ax=None):

    P = estimate_transition_probabilities(adata, delta_X, embedding, n_neighbors=n_neighbors, n_jobs=n_jobs)
    V_simulated = project_probabilities(P, layout_embedding, normalize=normalize)

    grid_scale = 10 * grid_scale / np.mean(abs(np.diff(layout_embedding)))
    print(grid_scale)
    get_grid_points = lambda min_val, max_val: np.linspace(min_val, max_val, 
                                                           int((max_val - min_val + 1) * grid_scale))

    grid_x = get_grid_points(np.min(layout_embedding[:, 0]), np.max(layout_embedding[:, 0]))
    grid_y = get_grid_points(np.min(layout_embedding[:, 1]), np.max(layout_embedding[:, 1]))
    grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
    size_x, size_y = len(grid_x), len(grid_y)
    
    vector_field = np.zeros((size_x, size_y, 2))

    x_thresh = (grid_x[1] - grid_x[0]) / 2
    y_thresh = (grid_y[1] - grid_y[0]) / 2

    get_neighborhood = lambda grid_point, layout_embedding: np.where(
        (np.abs(layout_embedding[:, 0] - grid_point[0]) <= x_thresh) &  
        (np.abs(layout_embedding[:, 1] - grid_point[1]) <= y_thresh)   
    )[0]

    for idx, grid_point in tqdm(enumerate(grid_points), desc='Computing vectors', total=len(grid_points)):

        # Get average vector within neighborhood
        indices = get_neighborhood(grid_point, layout_embedding)
        if len(indices) <= 0:
            continue
        nbr_vector = np.mean(V_simulated[indices], axis=0)
        nbr_vector *= len(indices)       # upweight vectors with lots of cells
            
        grid_idx_x, grid_idx_y = np.unravel_index(idx, (size_x, size_y))
        vector_field[grid_idx_x, grid_idx_y] = nbr_vector
    
    vector_field = vector_field.reshape(-1, 2)
    
    ### for testing, delete or save properly later
    # adata.uns['grid_points'] = grid_points
    # adata.uns['vector_field'] = vector_field
    adata.uns['nn_transition_P'] = P
    # adata.uns['V_simulated'] = V_simulated
    ###

    vector_scale = vector_scale / np.max(vector_field)
    vector_field *= vector_scale
    if annot is None:
        background = None
    else:
        background = {
            'X': layout_embedding[:, 0], 
            'Y': layout_embedding[:, 1], 
            'annot': list(adata.obs[annot]),
        }

    plot_quiver(grid_points, vector_field, background=background, ax=ax)

def distance_shift(adata, annot, ax=None):

    celltypes = sorted(adata.obs[annot].unique())
    ct_idxs = {ct: np.where(adata.obs[annot] == ct)[0] for ct in celltypes}
    delta_X = adata.layers['delta_X']

    ct_deltas = {ct: np.mean(delta_X[idx]) for ct, idx in ct_idxs.items()}

    # Plot distance shift
    sns.barplot(
        y=list(ct_deltas.keys()), x=list(ct_deltas.values()), ax=ax, 
        hue=celltypes, hue_order=celltypes)
    ax.set_title('Average Change in Count per Cell Type')
    ax.set_xlabel('Average Change in Count')
    ax.set_ylabel('Cell Type')
    
    return ax

def contour_shift(adata_train, gene, annot, seed=1334, ax=None):

    # Load data
    perturbed = adata_train.layers['simulated_count']
    gex = adata_train.layers['imputed_count']

    # Create UMAP embeddings
    reducer = umap.UMAP(random_state=seed, n_neighbors=50, min_dist=1.0, spread=5.0)
    X = np.vstack([gex, perturbed])
    umap_coords = reducer.fit_transform(X)

    # Split coordinates back into WT and KO
    n_wt = gex.shape[0]
    wt_umap = umap_coords[:n_wt]
    ko_umap = umap_coords[n_wt:]

    # Create UMAP visualization
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Plot cell type scatter points with custom styling
    sns.scatterplot(
        x=wt_umap[:,0], 
        y=wt_umap[:,1],
        hue=adata_train.obs[f'{annot}'].values,
        alpha=0.5,
        s=20,
        style=adata_train.obs[f'{annot}'].values,
        ax=ax,
        markers=['o', 'X', '<', '^', 'v', 'D', '>'],
    )

    # Add density contours for WT and KO
    for coords, label, color in [(wt_umap, 'WT', 'grey'), 
                                (ko_umap, 'KO', 'black')]:
        sns.kdeplot(
            x=coords[:,0],
            y=coords[:,1], 
            levels=8,
            alpha=1,
            linewidths=2,
            label=label,
            color=color,
            ax=ax,
            legend=True
        )

    # Style the plot
    ax.set_title(f'Cell Identity Shift from {gene} KO', pad=20, fontsize=12)
    ax.set_xlabel('UMAP 1', labelpad=10)
    ax.set_ylabel('UMAP 2', labelpad=10)
    ax.legend(ncol=1, loc='upper left', frameon=False)

    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
