import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import json

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import networkx as nx 
from networkx.algorithms import bipartite

from .oracles import SpaceTravLR
from .beta import BetaFrame
from .gene_factory import GeneFactory
from .models.parallel_estimators import get_filtered_df, create_spatial_features


class Visionary(GeneFactory):
    def __init__(self, ref_adata, test_adata, ref_json_path, 
                 prematching, matching_annot='cell_type', subsample=None):

        with open(ref_json_path, 'r') as f:
            params = json.load(f)

        super().__init__(adata=test_adata, 
                         models_dir=params['save_dir'], 
                         annot=params['annot'], 
                         radius=params['radius'], 
                         contact_distance=params['contact_distance'])

        self.ref_adata = ref_adata
        self.matching_annot = matching_annot

        # make annot (cell_type_int) match for ref and test adata
        ct_int_mapping = {
            matching_label: i for matching_label, i in self.ref_adata.obs[[self.matching_annot, self.annot]].value_counts().index}
        self.adata.obs[self.annot] = self.adata.obs[self.matching_annot].map(ct_int_mapping)

        self.matching = prematching
        self.adata.obs['reference_cell'] = prematching['reference_cell'].reindex(
            self.adata.obs.index, axis=0
        ).values
            
        self.reformat()
        self.compute_betas(subsample=subsample)

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
    
        



    

    
   