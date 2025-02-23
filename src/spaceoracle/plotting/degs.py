import numpy as np 
import pandas as pd 
import scanpy as sc 

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import os


def get_degs(adata, ko, method=None, show=10, ko_col='status', save_path=False):
    '''
    Get DEGs for a given KO
    @param method: method to use for DEG analysis
    @param show: number of DEGs to show in bar chart
    '''

    # Remove ko gene from adata
    # adata = adata[:, adata.var_names != ko].copy()

    sc.tl.rank_genes_groups(adata, ko_col, groups=['ko'], reference='wt', method=method)

    degs_df = sc.get.rank_genes_groups_df(adata, group='ko')
    degs_df.set_index('names', inplace=True)

    degs_df = degs_df.dropna(subset=['logfoldchanges'])

    degs_df['abs_lfc'] = degs_df['logfoldchanges'].abs()
    degs_df.sort_values('abs_lfc', ascending=False, inplace=True)
    degs_df = degs_df[degs_df['pvals_adj'] < 0.05]

    # Rank genes based on lfc
    degs_df['rank'] = degs_df['abs_lfc'].rank(ascending=False)

    try:
        if show: 
            show = min(show, degs_df.shape[0])

            plt.figure(figsize=(10, 8))
            colors = ['red' if lfc > 0 else 'blue' for lfc in degs_df['logfoldchanges'].head(show)]
            plt.barh(degs_df.head(show).index, degs_df['logfoldchanges'].head(show), color=colors)
            plt.xlabel("Log Fold Change")
            plt.ylabel("Genes")
            plt.title(f"{ko} Perturbation DEGs")
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
            plt.show()
            plt.close()
    except Exception as e:
        print(f'{ko} failed with error: {e}')

    return degs_df


def plot_corr(adata, nt_co, co, nt_st, pred, ko, genes=None, title='Change in Gene Expression Correlation', save_path=False):
    '''
    Plot change in gene expression correlation. 
    '''
    
    if genes is None:
        genes = list(adata.var_names)
    genes = [gene for gene in genes if gene != ko]
    tot_genes = len(genes.copy())

    co_nt = nt_co[genes].mean(axis=0)
    co_gex = co[genes].mean(axis=0).copy()
    st_nt = nt_st[genes].mean(axis=0)
    pred_gex = pred[genes].mean(axis=0).copy()

    # Get difference in gene expression from KO
    co_gex = co_gex - co_nt
    pred_gex = pred_gex - st_nt

    # Separate genes that were increased/ decreased from KO
    pos_genes = pred_gex[pred_gex > 0].index
    neg_genes = pred_gex[pred_gex < 0].index

    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(co_gex[pos_genes], pred_gex[pos_genes], alpha=0.5, color='red', label='Increased')
    plt.scatter(co_gex[neg_genes], pred_gex[neg_genes], alpha=0.5, color='blue', label='Decreased')
    plt.legend()
    
    plt.plot([co_gex.min(), co_gex.max()], [co_gex.min(), co_gex.max()], 'r--')
    spearman_corr, _ = spearmanr(co_gex, pred_gex)
    plt.title(f'Comparison of Predicted Delta GEX from {ko} KO \nSpearman Corr: {spearman_corr:.2f} ({len(genes)} genes)')
    plt.xlabel(f"CellOracle")
    plt.ylabel(f"SpaceTravLR")
    plt.grid(alpha=0.3)

    for i, gene in enumerate(genes):
        max_lfc = max(co_gex.max(), pred_gex.max())
        if abs(co_gex[gene]) >= 0.15 * max_lfc or abs(pred_gex[gene]) >= 0.15 * max_lfc:
            plt.annotate(gene, (co_gex[gene], pred_gex[gene]), fontsize=8, alpha=0.7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()






