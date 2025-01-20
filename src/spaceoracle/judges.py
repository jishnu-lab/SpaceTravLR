import numpy as np 
import pandas as pd 
import scanpy as sc 
import matplotlib.pyplot as plt
from numba import jit

from .plotting.degs import get_degs, plot_corr
from .plotting.sankey import plot_pysankey, get_macrostates
from .plotting.niche import get_demographics

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
    def plot_demographic_change(adata, delta_X, annot, celltype, radius=150, n_neighbors=10, nn_transitions=10, save_path=False):
        demo_df = get_demographics(adata, annot, radius=radius)

        ct_idxs = np.where(adata.obs[annot] == celltype)[0]
        demo_df = demo_df.iloc[ct_idxs]
        adata_ct = adata[ct_idxs, :]
        delta_X_ct = delta_X[ct_idxs]

        demo_annot = demo_df.idxmax(axis=1)
        embedding = demo_df.values

        adata_ct.obs['demographic'] = demo_annot

        Judge.plot_macrostate_change(
            adata_ct, delta_X_ct, 'demographic', embedding, 
            n_neighbors=n_neighbors, nn_transitions=nn_transitions, save_path=save_path
        )
        

    @staticmethod
    def plot_macrostate_change(adata, delta_X, annot, embedding, n_neighbors=200, nn_transitions=10, save_path=False):
        '''nn_transitions: number of cells to consider the transitions to'''
        sankey_df = get_macrostates(
            adata, 
            delta_X, 
            embedding, 
            annot, 
            n_neighbors=n_neighbors,
            nn_transitions=nn_transitions
        )

        delta_X_rndm = delta_X.copy()
        permute_rows_nsign(delta_X_rndm)

        sankey_df_rndm = get_macrostates(
            adata, 
            delta_X_rndm, 
            embedding, 
            annot, 
            n_neighbors=n_neighbors,
            nn_transitions=nn_transitions
        )

        sankey_df_rndm = sankey_df_rndm

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        plot_pysankey(sankey_df, ax=axs[0])
        plot_pysankey(sankey_df_rndm, ax=axs[1])

        axs[0].set_title('Celltype Transitions')
        axs[1].set_title('Randomized Control')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)

        plt.show()

    

# Cannibalized from CellOracle
@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])


