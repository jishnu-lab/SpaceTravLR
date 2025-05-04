import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import json
import enlighten

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
    '''
    A class for cross-predicting gene expression from a reference dataset to a test dataset.
    Reference and test datasets can differ in sample or modality, but should have similar
    spatial-resolution such that spots can be mapped to each other.
    '''
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

        self.matching = prematching.reindex(self.adata.obs.index, axis=0)
        self.adata.obs['reference_cell'] = self.matching['reference_cell'].values
            
        self.reformat()
        self.compute_betas(subsample=subsample)

    def reformat(self):
        # Create cell_thresholds for test adata
        cell_thresholds = self.ref_adata.uns['cell_thresholds'].loc[
            self.matching['reference_cell']
        ]
        self.adata.uns['cell_thresholds'] = cell_thresholds.set_index(
            pd.Index(self.adata.obs.index)
        )
    
    def compute_betas(self, subsample=None, float16=False):

        super().compute_betas(subsample=subsample, float16=float16, obs_names=self.ref_adata.obs_names)

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
    
        

class CyberBoss(Visionary):
    '''
    A class for cross-predicting gene expression from a reference dataset to a test dataset.
    Reference and test datasets can have different spatial-resolution and differ in context.
    '''

    def __init__(self, ref_adata, test_adata, ref_json_path, prematching, subsample=None):

        if 'imputed_count' not in test_adata.layers:
            # Don't want to use imputed counts for spots with potentially many cell types
            test_adata.layers['imputed_count'] = test_adata.layers['normalized_count']

        with open(ref_json_path, 'r') as f:
            params = json.load(f)

        GeneFactory.__init__(self, adata=test_adata, 
                         models_dir=params['save_dir'], 
                         annot=params['annot'], 
                         radius=params['radius'], 
                         contact_distance=params['contact_distance'])
    
        self.ref_adata = ref_adata
        self.matching = prematching.reindex(self.adata.obs.index, axis=0)
        self.adata.obs['reference_centroid'] = self.matching['nn_0']
        
        self.reformat()
        self.compute_betas(subsample=subsample)

    def compute_betas(self, subsample=None, float16=False):
        GeneFactory.compute_betas(
            self,
            subsample=subsample, 
            float16=float16, 
            obs_names=self.ref_adata.obs_names)

        # Take the median of all reference cell betas for each spot in test adata
        manager = enlighten.get_manager()
        pbar = manager.counter(
            desc=f'Reformatting betas genes',
            color='purple',
            justify=enlighten.Justify.CENTER,
            auto_refresh=True, total=len(self.beta_dict.data),
            width=30
        )

        test_beta_dict_data = {}

        for target_gene, betadata in self.beta_dict.data.items():
            betadata = self.beta_dict.data[target_gene]

            # df = self.matching.apply(lambda row: betadata.loc[row].median(axis=0), axis=1)
            df = self.matching.apply(lambda row: betadata.loc[row].mean(axis=0), axis=1)

            df.index = self.matching.index
            
            df = BetaFrame(df)
            df.modulator_gene_indices = [
                self.beta_dict.gene2index[g.replace('beta_', '')] for g in df.modulators_genes
            ]

            test_beta_dict_data[target_gene] = df 

            pbar.update()

        self.beta_dict.data = test_beta_dict_data

    
    def reformat(self):
        # Create cell_thresholds for test adata
        test_thresholds = []
        ref_thresholds = self.ref_adata.uns['cell_thresholds']

        for test_cell, row in self.matching.iterrows():
            test_thresholds.append(
                ref_thresholds.loc[self.matching.loc[test_cell]].sum(axis=0) 
            )

        test_thresholds = pd.DataFrame(test_thresholds, index=self.matching.index)
        test_thresholds.columns = ref_thresholds.columns
        test_thresholds.index.name = 'test_cell'
        self.adata.uns['cell_thresholds'] = test_thresholds
