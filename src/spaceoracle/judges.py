import numpy as np 
import pandas as pd 
import scanpy as sc 
import matplotlib.pyplot as plt
from numba import jit

from .plotting.degs import get_degs, plot_corr

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
    def get_expected_degs(adata, ko, method=None, show=100, ko_col='status', save_path=False):
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
    

# Cannibalized from CellOracle
@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])


