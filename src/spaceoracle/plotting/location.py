import numpy as np
import matplotlib.pyplot as plt


def get_cells_in_radius(point_coords, adata, annot=None, radius=100, cell_type=[], show=True):
    """ Get cells within a certain radius of a point. """
    cells_coords = adata.obsm['spatial']
    distances = np.linalg.norm(cells_coords - point_coords, axis=1)
    indices = np.where(distances < radius)[0]
    if len(cell_type) > 0:
        assert annot in adata.obs.columns, f'{annot} is not in adata.obs'
        indices = [i for i in indices if adata.obs[annot][i] in cell_type]
    
    if show:
        show_chosen_cells(adata, indices)

    return indices


def show_chosen_cells(adata, cell_idxs):
    """ Show chosen cells in a scatter plot. """
    plt.figure(figsize=(6,6))
    plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], c='lightgray', s=4, edgecolors='black', linewidth=0.1, label='Other cells')
    plt.scatter(adata.obsm['spatial'][cell_idxs, 0], adata.obsm['spatial'][cell_idxs, 1], c='blue', s=4, edgecolors='black', linewidth=0.1, label='Perturbed cells')
    plt.legend()
    plt.axis('off')
    plt.show()


def show_effect_distance(adata, annot, top_genes, point_coord, cutoff=1000):
    fig, axs = plt.subplots(len(top_genes), 1, figsize=(10, 6 * len(top_genes)))
    if len(top_genes) == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    for axs_idx, (ct, genes) in enumerate(top_genes.items()):
        ct_idxs = np.where(adata.obs[annot] == ct)[0]
        coords = adata.obsm['spatial'][ct_idxs]
        delta_X = adata.layers['delta_X'][ct_idxs]
        distances = np.linalg.norm(coords - point_coord, axis=1)

        for gene in genes:
            g = list(adata.var_names).index(gene)
            delta_X_g = delta_X[:, g]
            axs[axs_idx].scatter(distances, delta_X_g, label=gene, s=10, alpha=0.5, edgecolors='none')
        
        axs[axs_idx].set_title(f'Cell Type: {ct}', fontsize=14)
        axs[axs_idx].set_xlabel('Distance from perturbed cells', fontsize=12)
        axs[axs_idx].set_ylabel('GEX change', fontsize=12)
        axs[axs_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        axs[axs_idx].set_xlim([0, cutoff])
        axs[axs_idx].grid(True)

    plt.tight_layout()
    plt.show()

