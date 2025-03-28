import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
from sklearn.metrics.pairwise import cosine_similarity
import json

from .oracles import SpaceTravLR
from .beta import BetaFrame
from .gene_factory import GeneFactory
from .models.parallel_estimators import get_filtered_df, create_spatial_features


class Visionary(GeneFactory):
    def __init__(self, ref_adata, test_adata, ref_json_path, 
                 ref_embed=None, test_embed=None, subsample=None,
                 matching_annot=None, matching_method='banksy',
                 prematching=None):

        with open(ref_json_path, 'r') as f:
            params = json.load(f)

        super().__init__(adata=test_adata, 
                         models_dir=params['save_dir'], 
                         annot=params['annot'], 
                         radius=params['radius'], 
                         contact_distance=params['contact_distance'])

        self.ref_adata = ref_adata
        self.load_embeds(ref_embed, test_embed, method=matching_method)

        self.matching_annot = self.annot if matching_annot is None else matching_annot
        if prematching is None:
            self.match_embeddings()
        else:
            self.matching = prematching
            
        self.reformat()
        self.compute_betas(subsample=subsample)

    @staticmethod
    def compute_banksy_embed(adata, k_geom=15, max_m=1, nbr_weight_decay = "scaled_gaussian"):
        
        import sys
        sys.path.append('/ix/djishnu/alw399/SpaceOracle/src/Banksy_py')
        from banksy.main import median_dist_to_nearest_neighbour
        from banksy.initialize_banksy import initialize_banksy
        from banksy.embed_banksy import generate_banksy_matrix

        
        adata = adata.copy()
        adata.X = adata.layers['normalized_count']
        del adata.layers['raw_count']
        del adata.layers['imputed_count']

        # Find median distance to closest neighbours
        nbrs = median_dist_to_nearest_neighbour(adata, key = 'spatial')

        banksy_dict = initialize_banksy(
            adata,
            ('x', 'y', 'spatial'),
            k_geom,
            nbr_weight_decay=nbr_weight_decay,
            max_m=max_m,
            plt_edge_hist=True,
            plt_nbr_weights=True,
            plt_agf_angles=False, # takes long time to plot
            plt_theta=True,
        )


        # The following are the main hyperparameters for BANKSY
        # -----------------------------------------------------
        resolutions = [1.0]  # clustering resolution for UMAP
        pca_dims = [50]  # Dimensionality in which PCA reduces to
        lambda_list = [0.2]  # list of lambda parameters

        banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                            banksy_dict,
                                                            lambda_list,
                                                            max_m)

        return banksy_matrix.to_df()
        
    
    def compute_covet_embed(self, adata):
        import scenvi 

        # COVET uses the log normalized layer
        st_data = ad.AnnData(
            adata.layers['normalized_count'],
            var=adata.var,
            obsm=adata.obsm
        )
        st_data.layers['log'] = st_data.X
        
        # We want the SQRT to compare embeddings
        _, st_data.obsm['COVET_SQRT'], _ = scenvi.compute_covet(st_data)
        del st_data.layers['log']

        return st_data.obsm['COVET_SQRT']

    def compute_cnn_embed(self, adata):
        sp_maps = SpaceTravLR.imbue_adata_with_space(adata, annot=self.annot)
        spatial_features = create_spatial_features(
            adata.obsm['spatial'][:, 0], 
            adata.obsm['spatial'][:, 1], 
            adata.obs[self.annot], 
            adata.obs.index,
            radius=self.radius
        )
        raise NotImplementedError('CNN embeddings are not yet implemented')
        return None

    def load_embeds(self, ref_embed, test_embed, method='covet'):

        self.method = method

        if method == 'banksy':

            if ref_embed is None:
                ref_embed = self.compute_banksy_embed(self.ref_adata)
            if test_embed is None:
                test_embed = self.compute_banksy_embed(self.adata)
            
            test_embed = test_embed.reindex(ref_embed.columns, axis=1)
        
        elif method == 'cnn':

            if ref_embed is None:
                ref_embed = self.compute_cnn_embed(self.ref_adata)
            if test_embed is None:
                test_embed = self.compute_cnn_embed(self.adata)
       
        elif method == 'covet':

            if ref_embed is None:
                ref_embed = self.compute_covet_embed(self.ref_adata)
            if test_embed is None:
                test_embed = self.compute_covet_embed(self.adata)
        
            ref_embed = pd.DataFrame(
                ref_embed.reshape(ref_embed.shape[0], -1),
                index=self.ref_adata.obs.index,
                columns=[f'COVET_{i}' for i in range(ref_embed.shape[1]**2)]
            )
            test_embed = pd.DataFrame(
                test_embed.reshape(test_embed.shape[0], -1),
                index=self.adata.obs.index,
                columns=[f'COVET_{i}' for i in range(test_embed.shape[1]**2)]
            )
        
        else:
            raise ValueError('Please choose a valid method: banksy, cnn, or covet')

        self.ref_embed = ref_embed
        self.test_embed = test_embed

    @staticmethod
    def _match_cells_cosine(test_embed, ref_embed):

        cos_sim_matrix = cosine_similarity(test_embed, ref_embed)
        closest_indices = np.argmax(cos_sim_matrix, axis=1)
        matched_refs = ref_embed.iloc[closest_indices]
        matched = matched_refs.index

        return matched
    
    @staticmethod
    def _match_cells_difference(test_embed, ref_embed):

        diff_matrix = np.abs(
            test_embed.values.astype(np.float16)[:, None] - ref_embed.values.astype(np.float16))
        closest_indices = np.argmin(
            diff_matrix.sum(axis=2),    # sum along COVET embedding dimension
            axis=-1                     # find the most similar reference cell
        )
        matched_refs = ref_embed.iloc[closest_indices]
        matched = matched_refs.index

        return matched
    
    def match_cells(self, test_embed, ref_embed):
        if self.method == 'covet':
            return self._match_cells_difference(test_embed, ref_embed)
        elif self.method == 'banksy':
            return self._match_cells_cosine(test_embed, ref_embed)
        else:
            raise ValueError('Please choose a valid method')

    def match_embeddings(self):

        matching_annot = self.matching_annot
        celltypes = self.adata.obs[matching_annot].unique()

        test_cts = {ct: self.adata.obs[matching_annot] == ct for ct in celltypes}
        ref_cts = {ct: self.ref_adata.obs[matching_annot] == ct for ct in celltypes}
        test_cells = self.adata.obs.index

        matched = pd.DataFrame(index=test_cells , columns=['reference_cell'])
        for ct, ct_mask in test_cts.items():
            matched.loc[test_cells[ct_mask], 'reference_cell'] = self.match_cells(
                self.test_embed.loc[ct_mask], self.ref_embed[ref_cts[ct]]).values
            self.adata.obs.loc[ct_mask, 'reference_cell'] = matched.loc[test_cells[ct_mask], 'reference_cell'].values
        
        self.matching = matched
    
    def reformat(self):
        cell_thresholds = self.ref_adata.uns['cell_thresholds'].loc[
            self.matching['reference_cell']
        ]
        self.adata.uns['cell_thresholds'] = cell_thresholds.set_index(
            pd.Index(self.adata.obs.index)
        )
    
    def compute_betas(self, subsample=None, float16=False):
        
        super().compute_betas(subsample=subsample, float16=float16)

        self.beta_dict.data = {
            k: v.reindex(self.adata.obs['reference_cell'], axis=0)
                        .set_index(pd.Index(self.adata.obs.index))
            for k, v in self.beta_dict.data.items()
        }

    @staticmethod
    def load_betadata(gene, save_dir, matching):
        # return pd.read_parquet(f'{save_dir}/{gene}_betadata.parquet')
        betadata = BetaFrame.from_path(f'{save_dir}/{gene}_betadata.parquet')
        return betadata.reindex(matching['reference_cell'], axis=0).set_index(pd.Index(matching.index))

    def splash_betas(self, gene):
        rw_ligands = self.adata.uns.get('received_ligands')
        rw_tfligands = self.adata.uns.get('received_ligands_tfl')
        gene_mtx = self.adata.layers['imputed_count']
        cell_thresholds = self.adata.uns.get('cell_thresholds')
        
        if rw_ligands is None or rw_tfligands is None:
            rw_ligands = self._compute_weighted_ligands(
                gene_mtx, cell_thresholds, genes=self.ligands)
            rw_tfligands = self._compute_weighted_ligands(
                gene_mtx, cell_thresholds=None, genes=self.tfl_ligands)
            self.adata.uns['received_ligands'] = rw_ligands
            self.adata.uns['received_ligands_tfl'] = rw_tfligands

        filtered_df = get_filtered_df(
            counts_df=pd.DataFrame(
                gene_mtx, 
                index=self.adata.obs_names, 
                columns=self.adata.var_names
            ),
            cell_thresholds=cell_thresholds,
            genes=self.adata.var_names
        )[self.adata.var_names] 
        
        betadata = self.load_betadata(gene, self.save_dir, self.matching)
        
        return self._combine_gene_wbetas(
            rw_ligands, rw_tfligands, filtered_df, betadata)
    
        



    

    
   