import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from spaceoracle.plotting.shift import estimate_transition_probabilities

def quantify_correlations(adata, annot='cell_type'):
    celltypes = adata.obs[annot].unique()

    corr_mat = np.zeros((len(celltypes), len(celltypes)))

    for i, source_ct in enumerate(celltypes):
        source_idxs = adata.obs[annot] == source_ct
        
        for j, target_ct in enumerate(celltypes):
            target_idxs = adata.obs[annot] == target_ct
            
            corrs = np.corrcoef(
                adata.layers['imputed_count'][source_idxs], 
                adata.layers['simulated_count'][target_idxs]
            )
            x = source_idxs.sum()
            y = target_idxs.sum()
            cross_corr = corrs[:x, x:]
            orig_corr = corrs[:x, :y]
            corr_mat[i, j] = cross_corr.mean() - orig_corr.mean()
    
    heatmap_df = pd.DataFrame(corr_mat, index=celltypes, columns=celltypes)
    return heatmap_df


def plot_corr_heatmap(heatmap_df, title='Cross-correlation between imputed and simulated counts', save_path=None):

    # heatmap_df = sankey_df.pivot(index='source', columns='target', values='value').fillna(0)

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        heatmap_df, 
        cmap='Spectral', 
        annot=True, 
        cbar=True, 
        center=0,
        linewidths=.5, 
        linecolor='black'
    )
    plt.title(title)
    plt.xlabel('Perturbed Cells')
    plt.ylabel('Original Cells')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    