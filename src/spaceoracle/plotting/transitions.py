import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns 
import umap
import os

from .layout import plot_quiver, get_grid_layout
from .shift import *



def estimate_transitions_2D(adata, delta_X, embedding, layout_embedding, annot=None, normalize=True, 
n_neighbors=200, grid_scale=1, vector_scale=1, n_jobs=1, ax=None):
    P = estimate_transition_probabilities(adata, delta_X, embedding, n_neighbors=n_neighbors, n_jobs=n_jobs)
    P_null = estimate_transition_probabilities(adata, delta_X * 0, embedding, n_neighbors=n_neighbors, n_jobs=n_jobs)
    P = P - P_null
    V_simulated = project_probabilities(P, layout_embedding, normalize=normalize)

    grid_scale = 10 * grid_scale / np.mean(abs(np.diff(layout_embedding)))
    # print(grid_scale)

    # get_grid_points = lambda min_val, max_val: np.linspace(min_val, max_val, 
    #                                                        int((max_val - min_val + 1) * grid_scale))
    # grid_x = get_grid_points(np.min(layout_embedding[:, 0]), np.max(layout_embedding[:, 0]))
    # grid_y = get_grid_points(np.min(layout_embedding[:, 1]), np.max(layout_embedding[:, 1]))
    grid_x, grid_y = get_grid_layout(layout_embedding, grid_scale=grid_scale)
    
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


def distance_shift(adata, annot, ax=None, n_show=5, compare_ct=True, ct_interest=None, save_dir=False):
    '''
    compare_ct: if True, show the genes with the greatest delta difference between cell types
    ct_interest: cell type of interest to compare against all other cell types
    '''

    celltypes = sorted(adata.obs[annot].unique())
    ct_idxs = {ct: np.where(adata.obs[annot] == ct)[0] for ct in celltypes}

    delta_X = adata.layers['delta_X']
    ct_deltas = {ct: delta_X[idx] for ct, idx in ct_idxs.items()}
    ct_means = pd.DataFrame({ct: np.mean(vals, axis=0) for ct, vals in ct_deltas.items()})

    # Identify genes with the greatest difference in change between cell types
    if ct_interest is not None:
        assert ct_interest in celltypes, f'{ct_interest} not found in cell types'
        
        if compare_ct:
            minus_max = np.array(ct_means[ct_interest] - ct_means.drop(columns=ct_interest).max(axis=1))
            minus_min = np.array(ct_means[ct_interest] - ct_means.drop(columns=ct_interest).min(axis=1))
            gene_diffs = np.maximum(abs(minus_max), abs(minus_min))

        else:
            gene_diffs = np.array(abs(ct_means[ct_interest]))

    elif compare_ct:
        gene_diffs = np.array(ct_means.max(axis=1) - ct_means.min(axis=1))

    else:
        gene_diffs = np.array(abs(ct_means).max(axis=1))

    non_zero_count = np.count_nonzero(gene_diffs)
    if non_zero_count == 0:
        print(f'No change detected in {celltypes} gene expression')
        return None
    
    n_show = min(n_show, non_zero_count)
    top_gene_idxs = np.argsort(gene_diffs)[-n_show:]
    top_gene_labels = list(adata.var_names[top_gene_idxs])

    # Extract deltas for top genes
    top_ct_deltas = {ct: deltas[:, top_gene_idxs] for ct, deltas in ct_deltas.items()}

    # Prepare data for plotting
    plot_data = []
    for ct, deltas in top_ct_deltas.items():
        for gene_idx, gene_label in zip(range(deltas.shape[1]), top_gene_labels):
            plot_data.append(pd.DataFrame({
                'delta': deltas[:, gene_idx],
                'gene': gene_label,
                'ct': ct
            }))

    df = pd.concat(plot_data, ignore_index=True)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = sns.boxplot(
        x='gene', y='delta', hue='ct', 
        ax=ax, hue_order=celltypes,
        data=df
    )

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    ax.set_title('Average Gene Count Change per Cell Type')
    ax.set_ylabel('Average Change in Count')
    ax.set_xlabel('Gene')

    if save_dir:
        names = '_'.join(sorted(top_ct_deltas.keys()))
        joint_name = f'delta_{names}.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, joint_name), bbox_inches='tight')

    return top_gene_labels


def contour_shift(adata_train, title, annot, seed=1334, ax=None, perturbed=None):

    # Load data
    if perturbed is None:
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
    ax.set_title(title, pad=20, fontsize=12)
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
