import numpy as np 
import pandas as pd 
import scanpy as sc 
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .plotting.degs import get_degs, plot_corr
from .plotting.metacells import get_SEACells_assigns

class Judge():
    def __init__(self, adata, annot):
        self.adata = adata
        self.annot = annot
    
    def create_sim_adata(self, ctrl_count, ko_count, ko_obs_col='status'):
        '''Create simulated adata to use built in scanpy DEG functions'''
        sim_adata = sc.AnnData(
            X=np.concatenate([ctrl_count, ko_count], axis=0),
            obs=pd.DataFrame(
                data={
                    ko_obs_col: ['wt'] * ctrl_count.shape[0] + ['ko'] * ko_count.shape[0],
                    self.annot: list(self.adata.obs[self.annot].values) * 2
                
                },
                index=[f'cell{i}' for i in range(ctrl_count.shape[0] + ko_count.shape[0])]
            ),
            var=pd.DataFrame(
                data={'gene': self.adata.var_names},
                index=self.adata.var_names
            )
        )

        sim_adata.obs[ko_obs_col] = sim_adata.obs[ko_obs_col].astype('category')

        # Perform PCA
        sc.tl.pca(sim_adata)
        sim_adata.obsm['X_pca'] = sim_adata.obsm['X_pca']

        # Perform UMAP
        sc.pp.neighbors(sim_adata, use_rep='X_pca')
        sc.tl.umap(sim_adata)
        sim_adata.obsm['X_umap'] = sim_adata.obsm['X_umap']

        return sim_adata
    
    @staticmethod
    def get_expected_degs(adata, ko, method=None, show=10, ko_col='status', save_path=False):
        '''
        @param method: method to use for DEG analysis
        @param show: number of DEGs to show in bar chart
        '''
        degs_df = get_degs(adata, ko, method=method, show=show, ko_col=ko_col, save_path=save_path)
        return degs_df
    
    def plot_delta_corr(self, nt, co, pred, ko, genes=None, title='Change in Gene Expression Correlation', save_path=False):
        '''
        Plot change in gene expression correlation. 
        @param nt: control count matrix (imputed count)
        @param co: CO KO count (CO simulated count)
        @param pred: ST KO count (ST simulated count)
        '''
        plot_corr(
            self.adata, nt_st=nt, nt_co=nt, co=co, pred=pred,
            ko=ko, genes=genes, title=title, save_path=save_path
        )

    @staticmethod
    def get_macrostates(sim_adata, ko_col='status', n_SEACells=90, build_kernel_on = 'X_pca', n_waypoint_eigs = 10, show=False):
        '''Get SEACEll soft assignments'''
        print('Getting macrostates...')
        wt_adata = sim_adata[sim_adata.obs[ko_col] == 'wt']
        wt_labels, wt_weights = get_SEACells_assigns(
            wt_adata, n_SEACells=n_SEACells, build_kernel_on=build_kernel_on, n_waypoint_eigs=n_waypoint_eigs, show=show)

        print('Getting KO macrostates...')
        ko_adata = sim_adata[sim_adata.obs[ko_col] == 'ko']
        ko_labels, ko_weights = get_SEACells_assigns(
            ko_adata, n_SEACells=n_SEACells, build_kernel_on=build_kernel_on, n_waypoint_eigs=n_waypoint_eigs, show=show)

        data = {
            'wt': {
                'labels': wt_labels,
                'weights': wt_weights
            },
            'ko': {
                'labels': ko_labels,
                'weights': ko_weights
            }
        }
        return data
    
    @staticmethod
    def get_macrostate_change(sim_adata, seacells_data, annot):
        cell2ct = {cell : ct for cell, ct in zip(sim_adata.obs_names, list(sim_adata.obs[annot]) * 2)}
        
        seacells_data['wt']['labels'].replace(cell2ct, inplace=True)
        seacells_data['ko']['labels'].replace(cell2ct, inplace=True)

        celltypes = sim_adata.obs[annot].unique()

        sankey_data = {'source': [], 'target': [], 'value': []}

        for wt_ct in celltypes:
            mask_wt = (seacells_data['wt']['labels'] == wt_ct).values

            for ko_ct in celltypes:
                mask_ko = (seacells_data['ko']['labels'] == ko_ct).values

                # Compute how much of the wt cell type goes to ko cell type
                shifted_val = np.maximum( 
                    0, seacells_data['ko']['weights'][mask_wt & mask_ko] - seacells_data['wt']['weights'][mask_wt & mask_ko]
                ).sum()

                sankey_data['source'].append(wt_ct)
                sankey_data['target'].append(ko_ct)
                sankey_data['value'].append(shifted_val)

        # Convert to DataFrame
        sankey_df = pd.DataFrame(sankey_data)
        return sankey_df



            

