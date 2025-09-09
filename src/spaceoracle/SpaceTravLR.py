#                                _..._
#                            .'     '.      _
#                            /    .-""-\   _/ \
#                        .-|   /:.   |  |   |
#                        |  \  |:.   /.-'-./
#                        | .-'-;:__.'    =/
#                        .'=  *=|     _.='
#                        /   _.  |    ;
#                        ;-.-'|    \   |
#                        /   | \    _\  _\
#                        \__/'._;.  ==' ==\
#                                \    \   |
#                                /    /   /
#                                /-._/-._/
#                                \   `\  \
#                                `-._/._/

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import commot as ct
import sys 
from collections import defaultdict
from tqdm import tqdm
import anndata as ad
import pickle

class SpaceShip:
    def __init__(self, adata, annot='cell_type'):
        from .oracles import BaseTravLR
        from .tools.utils import scale_adata, is_mouse_data
        from .tools.network import encode_labels
        
        assert isinstance(adata, ad.AnnData)
        assert annot in adata.obs.columns
        assert 'spatial' in adata.obsm
        assert 'normalized_count' in adata.layers
        
        self.species = 'mouse' if is_mouse_data(adata) else 'human'
        
        adata = scale_adata(adata)
        
        adata.obs['cell_type_int'] = adata.obs[annot].apply(
            lambda x: encode_labels(adata.obs[annot], reverse_dict=True)[x])
        
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

        BaseTravLR.impute_clusterwise(
            adata, 
            annot=annot, 
            layer='normalized_count', 
            layer_added='imputed_count'
        )
        
        self.adata = adata
        self.annot = annot
        
    def run_celloracle(self):
        sys.path.append('../')
        import celloracle as co
        
        
        adata = self.adata
        
        oracle = co.Oracle()
        adata.X = adata.layers["raw_count"].copy()
        oracle.import_anndata_as_raw_count(
            adata=adata,
            cluster_column_name=self.annot,
            embedding_name="X_umap"
        )
        oracle.pcs = [True]
        oracle.k_knn_imputation = 1
        oracle.knn = 1
        
        if self.species == 'human':
            base_GRN = co.data.load_human_promoter_base_GRN()
        else:
            base_GRN = co.data.load_mouse_promoter_base_GRN()

            
        oracle.import_TF_data(TF_info_matrix=base_GRN)
        
        links = oracle.get_links(
            cluster_name_for_GRN_unit=self.annot, 
            alpha=5,
            verbose_level=1
        )
        
        links.filter_links()
        oracle.get_cluster_specific_TFdict_from_Links(links_object=links)

        
        self.links = links.links_dict
    
    def run_commot(self, radius=350):
        from .tools.network import expand_paired_interactions
        from .tools.network import get_cellchat_db
        from .models.parallel_estimators import init_received_ligands
        import commot as ct
        
        adata = self.adata
        
        df_ligrec = get_cellchat_db(self.species) 
        df_ligrec['name'] = df_ligrec['ligand'] + '-' + df_ligrec['receptor']
        
        expanded = expand_paired_interactions(df_ligrec)
        genes = set(expanded.ligand) | set(expanded.receptor)
        genes = list(genes)

        expanded = expanded[
            expanded.ligand.isin(adata.var_names) & expanded.receptor.isin(adata.var_names)]
        
        adata.X = adata.layers['normalized_count']
        
        ct.tl.spatial_communication(adata,
            database_name='user_database', 
            df_ligrec=expanded, 
            dis_thr=radius, 
            heteromeric=False
        )
        
        expanded['rename'] = expanded['ligand'] + '-' + expanded['receptor']
            
        for name in tqdm(expanded['rename'].unique()):
            ct.tl.cluster_communication(
                adata, 
                database_name='user_database', 
                pathway_name=name, 
                clustering='cell_type',
                random_seed=12, 
                n_permutations=100
            )
            
        data_dict = defaultdict(dict)

        for name in expanded['rename']:
            data_dict[name]['communication_matrix'] = adata.uns[
                f'commot_cluster-cell_type-user_database-{name}']['communication_matrix']
            data_dict[name]['communication_pvalue'] = adata.uns[
                f'commot_cluster-cell_type-user_database-{name}']['communication_pvalue']

        import pickle
        with open('/tmp/communication.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            
            
        info = data_dict
        
        def get_sig_interactions(value_matrix, p_matrix, pval=0.3):
            p_matrix = np.where(p_matrix < pval, 1, 0)
            return value_matrix * p_matrix
        
        interactions = {}
        for lig, rec in tqdm(zip(expanded['ligand'], expanded['receptor'])):
            name = lig + '-' + rec
            if name in info.keys():
                value_matrix = info[name]['communication_matrix']
                p_matrix = info[name]['communication_pvalue']
                sig_matrix = get_sig_interactions(value_matrix, p_matrix)
                if sig_matrix.sum().sum() > 0:
                    interactions[name] = sig_matrix
                    
                    
        # create cell x gene matrix
        ct_masks = {ct: adata.obs[self.annot] == ct for ct in adata.obs[self.annot].unique()}
        df = pd.DataFrame(index=adata.obs_names, columns=genes)
        df = df.fillna(0)
        for name in tqdm(interactions.keys(), total=len(interactions)):
            lig, rec = name.rsplit('-', 1)
            tmp = interactions[name].sum(axis=1)
            for ct, val in zip(interactions[name].index, tmp):
                df.loc[ct_masks[ct], lig] += tmp[ct]
            tmp = interactions[name].sum(axis=0)
            for ct, val in zip(interactions[name].columns, tmp):
                df.loc[ct_masks[ct], rec] += tmp[ct]
                
        perc_filtered = np.where(df > 0, 1, 0).sum().sum() / (df.shape[0] * df.shape[1])      
        print('Percentage of LR filtered using celltype specificity:', perc_filtered)
        
        df.to_parquet('/tmp/LRs.parquet')
        
        adata.uns['cell_thresholds'] = df.copy()
        
        adata = init_received_ligands(
            adata, 
            radius=300, 
            contact_distance=50, 
            cell_threshes=df
        )
        
        keys = list(adata.obsm.keys())
        for key in keys:
            if 'commot' in key:
                del adata.obsm[key]
                
        keys = list(adata.uns.keys())
        for key in keys:
            if 'commot' in key:
                del adata.uns[key]
                
        keys = list(adata.obsp.keys())
        for key in keys:
            if 'commot' in key:
                del adata.obsp[key]
                
                
        self.adata = adata.copy()
        
    def run_spacetravlr(self):
        from .oracles import SpaceTravLR
        from .tools.network import RegulatoryFactory
        from .gene_factory import GeneFactory
        
        
        base_dir = '/tmp/'
        adata = self.adata

        co_grn = RegulatoryFactory(
            links=self.links,
            annot='cell_type_int'
        )
        
        
        star = SpaceTravLR(
            adata=adata,
            annot='cell_type_int', 
            max_epochs=150, 
            learning_rate=5e-3, 
            spatial_dim=64,
            batch_size=512,
            grn=co_grn,
            radius=400,
            contact_distance=50,
            save_dir=base_dir + 'lasso_runs'
        )

        star.run()
        
        
        self.gf = GeneFactory.from_json(
            adata=star.adata, 
            json_path=star.save_dir + '/run_params.json', 
        )

        



        
        








