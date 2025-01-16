import numpy as np 
import pandas as pd 
import scanpy as sc 
import json, os

import matplotlib.pyplot as plt
import seaborn as sns 

from scipy.stats import spearmanr

class Judge():
    def __init__(self, adata):
        self.adata = adata
    
    def create_sim_adata(self, ctrl_count, ko_count, ko_obs_col='status'):
        '''Create simulated adata to evaluate performance'''
        sim_adata = sc.AnnData(
            X=np.concatenate([ctrl_count, ko_count], axis=0),
            obs=pd.DataFrame(
                data={
                    ko_obs_col: ['wt'] * ctrl_count.shape[0] + ['ko'] * ko_count.shape[0]
                },
                index=[f'cell{i}' for i in range(ctrl_count.shape[0] + ko_count.shape[0])]
            ),
            var=pd.DataFrame(
                data={'gene': self.adata.var_names},
                index=self.adata.var_names
            )
        )

        sim_adata.obs[ko_obs_col] = sim_adata.obs[ko_obs_col].astype('category')

        return sim_adata
    
    @staticmethod
    def get_expected_degs(adata, ko, method=None, show=10, ko_col='status', save_path=False):
        '''
        Get DEGs for a given KO
        @param method: method to use for DEG analysis
        @param show: number of DEGs to show in bar chart
        '''

        # Remove ko gene from adata
        # adata = adata[:, adata.var_names != ko].copy()

        sc.tl.rank_genes_groups(adata, ko_col, groups=['ko'], reference='wt', method=method)
        # sc.pl.rank_genes_groups(self.adata, n_genes=20, sharey=False)

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
    

    def plot_delta_corr(self, nt_co, co, nt_st, pred, ko, genes=None, title='Change in Gene Expression Correlation', save_path=False):
        '''
        Plot change in gene expression correlation. 
        '''
        
        if genes is None:
            genes = list(self.adata.var_names)
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
        plt.figure(figsize=(10, 6))
        plt.scatter(co_gex[pos_genes], pred_gex[pos_genes], alpha=0.5, color='red', label='Increased')
        plt.scatter(co_gex[neg_genes], pred_gex[neg_genes], alpha=0.5, color='blue', label='Decreased')
        plt.legend()
        
        plt.plot([co_gex.min(), co_gex.max()], [co_gex.min(), co_gex.max()], 'r--')
        spearman_corr, _ = spearmanr(co_gex, pred_gex)
        plt.title(f'Comparison of Predicted Delta GEX from {ko} KO \nSpearman Corr: {spearman_corr:.2f}')
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

    
    def evaluate_preds(nt, gt, pred, top_degs, title, save_path=False, show=True):
        """
        Density plot to compare predictions against wt, co, and st predictions
        """
        nt_count = nt[top_degs]
        gt_count = gt[top_degs]
        sim_count = pred[top_degs]

        # Create plot layout
        num_genes = len(top_degs)
        fig, axs = plt.subplots(num_genes, 1, figsize=(10, 6 * num_genes), sharex=True)

        if num_genes == 1:
            axs = [axs]

        # Plot density plots
        for i, gene in enumerate(top_degs):
            sns.kdeplot(data=gt_count[gene], ax=axs[i], color='blue', label='CRISPRi', alpha=0.3)            
            
            if pred.shape[0] == 1:
                axs[i].axvline(x=sim_count[gene].mean(), color='orange', linestyle='--', linewidth=1, label='Predicted')
            else: 
                sns.kdeplot(data=sim_count[gene], ax=axs[i], color='orange', label='Predicted', alpha=0.3)
            
            sns.kdeplot(data=nt_count[gene], ax=axs[i], color='green', label='Non-targeting', alpha=0.3)

            axs[i].legend()
            axs[i].set_title(gene)
        
        fig.suptitle(title)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
        if show:
            plt.show()

        plt.close()

