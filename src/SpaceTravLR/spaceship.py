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


import os
import sys 
import pickle
import functools
import time

# import jscatter
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import enlighten

from datetime import timedelta
from tqdm import tqdm
from collections import defaultdict
from simple_slurm import Slurm  # pyright: ignore[reportMissingImports]

from SpaceTravLR.tools.network import expand_paired_interactions, get_cellchat_db

from enum import Enum

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class Status(Enum):
    BORN        =   "Newly born"
    BORED       =   "Ready but not doing anything"
    RUNNING     =   "Running"
    SUCCESS     =   "Completed everything gracefully"
    FUBAR       =   "F- Up Beyond Repair"

""" 
default output directory is 'output' 
'output/input_data' stores all the inputs
'output/logs' stores logs
'output/betadata'  stores the spatial gene-gene networks

methods with trailing underscores have side-effects but return Nothing
code philosophy is to fail early and loudly
"""


def catch_and_retry(retry=1):
    def wrapper(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            for i in range(0, retry):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    print(Status.FUBAR)
                    raise e
                    time.sleep(i+1)
        return inner
    return wrapper

catch_errors = catch_and_retry(retry=1) #alias

class SpaceShip:
    """
    SpaceShip is the main entry point for the SpaceTravLR analysis pipeline.
    It manages the data, directory structure, and execution of the various
    steps involved in spatial gene regulatory network inference and perturbation.

    Parameters
    ----------
    name : str, optional
        Name of the project/analysis, by default 'AlienTissue'.
    outdir : str, optional
        Path to the output directory where results will be saved, by default './output'.
    """
    def __init__(self, name: str = 'AlienTissue', outdir: str = './output'):
        self.name = name
        self.outdir = outdir.rstrip("/\\")
        self.manager = None
        self.status_bar = None
        
        self.status = Status.BORN
     
    @catch_errors
    def process_adata_(self, adata: ad.AnnData, annot: str = 'cell_type'):
        """
        Preprocesses the AnnData object for SpaceTravLR analysis.
        
        This method checks for required fields, normalizes data if necessary,
        computes PCA/Neighbors/UMAP if missing, and imputes gene expression
        if needed. It saves the processed AnnData to the output directory.

        Parameters
        ----------
        adata : ad.AnnData
            The AnnData object containing the spatial transcriptomics data.
        annot : str, optional
            The column name in `adata.obs` containing cell type annotations,
            by default 'cell_type'.
        """
        
        from .oracles import BaseTravLR
        from .tools.utils import scale_adata, is_mouse_data
        from .tools.network import encode_labels
        
        if self.status_bar:
            self.status_bar.update('📊 Processing AnnData: Validating input...')
        
        assert isinstance(adata, ad.AnnData)
        assert annot in adata.obs.columns
        assert 'spatial' in adata.obsm
        
        adata = adata.copy()
        
        self.species = 'mouse' if is_mouse_data(adata) else 'human'        
        adata = scale_adata(adata)
        
        adata.obs['cell_type_int'] = adata.obs[annot].apply(
            lambda x: encode_labels(adata.obs[annot], reverse_dict=True)[x])
        
        if 'X_umap' not in adata.obsm:
            if self.status_bar:
                self.status_bar.update('📊 Processing AnnData: Computing PCA, neighbors, and UMAP...')
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            
        if 'imputed_count' not in adata.layers:
            if 'normalized_count' not in adata.layers:
                if adata.X.max() > 100:
                    sc.pp.log1p(adata)
            
            adata.layers['normalized_count'] = adata.X.copy()

            BaseTravLR.impute_clusterwise(
                adata, 
                annot=annot, 
                layer='normalized_count', 
                layer_added='imputed_count'
            )
            
            # del adata.layers['normalized_count']
        
        self.annot = annot
        
        if self.status_bar:
            self.status_bar.update('📊 Processing AnnData: Saving processed data...')
        
        adata.obs.index = adata.obs.index.astype(str)
        adata.var.index = adata.var.index.astype(str)

        adata.write_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        self.adata = adata
        
        if self.status_bar:
            self.status_bar.update('✅ Processing AnnData: Complete')
            
    def interactive_select(self, adata, size=10, annot='cell_type', mode='spatial'):
        """
        Launches an interactive scatter plot for selecting cells.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object.
        size : int, optional
            Point size, by default 10.
        annot : str, optional
            Color by annotation, by default 'cell_type'.
        mode : str, optional
            'spatial' or 'umap', by default 'spatial'.

        Returns
        -------
        jscatter.Scatter
            Interactive scatter plot widget.
        """
        import jscatter
        datadf_with_umap = adata.to_df().join(adata.obs).join(
            pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
                )
        datadf_with_umap['umapX'] = adata.obsm['X_umap'][:,0]
        datadf_with_umap['umapY'] = adata.obsm['X_umap'][:,1]

        config = {'height': 800, 'width': 800, 'size': size}

        if mode == 'spatial':
            scatter = jscatter.Scatter(data=datadf_with_umap, x='x', y='y', **config).color(by=annot).legend(True)
        elif mode == 'umap':
            scatter = jscatter.Scatter(data=datadf_with_umap, x='umapX', y='umapY', **config).color(by=annot).legend(True)
        else:
            raise ValueError(f"Invalid mode: {mode}")
            
        return scatter    
        
    def load_base_cell_thresholds(self) -> pd.DataFrame:
        df_ligrec = get_cellchat_db(self.species) 
        df_ligrec['name'] = df_ligrec['ligand'] + '-' + df_ligrec['receptor']
        expanded = expand_paired_interactions(df_ligrec)
        genes = set(expanded.ligand) | set(expanded.receptor)
        genes = list(genes)

        return pd.DataFrame(
            columns=genes, 
            index=self.adata.obs_names
        ).fillna(1).astype(int)
        
    @staticmethod 
    def load_base_GRN(species) -> pd.DataFrame:
        assert species in ['human', 'mouse']

        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'SpaceTravLR_data', f'{species}_base_grn.parquet')
        df = pd.read_parquet(data_path)

        # tf_columns = [col for col in df.columns if col not in ['peak_id', 'gene_short_name']]
        # df = df.melt(
        #     id_vars=['gene_short_name'], 
        #     value_vars=tf_columns,
        #     var_name='source', 
        #     value_name='link').query(
        #         'link == 1')[['source', 'gene_short_name']].rename(
        #             columns={'gene_short_name': 'target'})
            
        # df['coef_mean'] = 1
        # df['coef_abs'] = 1
        # df['p'] = 1e-5
        # df['-logp'] = 5

        return df
            
    
    @catch_errors  
    def run_celloracle_(self, alpha=5, base_GRN=None):
        """
        Runs CellOracle to infer the base Gene Regulatory Network (GRN).
        
        It constructs a cluster-specific GRN based on the base network structure
        and the expression data in the AnnData object.

        Parameters
        ----------
        alpha : int, optional
            Regularization parameter for the model, by default 5.
        """
        if self.status_bar:
            self.status_bar.update('Building base GRN...')
        
        import celloracle_tmp as co

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
        
        if not base_GRN:
            base_GRN = self.load_base_GRN(self.species)

        oracle.import_TF_data(TF_info_matrix=base_GRN)
        
        if self.status_bar:
            self.status_bar.update('Computing & filtering TF links...')
        
        links = oracle.get_links(
            cluster_name_for_GRN_unit=self.annot, 
            alpha=alpha,
            verbose_level=0
        )

        links.filter_links()
        oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
        
        self.links = links.links_dict
        
        with open(f'{self.outdir}/input_data/celloracle_links.pkl', 'wb') as f:
            pickle.dump(links.links_dict, f)
    
    @catch_errors
    def run_commot_(self, radius=350):
        """
        Runs COMMOT to infer spatial cell-cell communication.
        
        This method identifies ligand-receptor interactions and computes their
        spatial communication scores. It also computes received ligand signals
        for each cell.

        Parameters
        ----------
        radius : int, optional
            Spatial radius for communication in microns (or coordinate units),
            by default 350.
        """
        from .tools.network import expand_paired_interactions
        from .tools.network import get_cellchat_db
        from .models.parallel_estimators import init_received_ligands
        import commot as ct
        
        adata = self.adata

        if 'cell_thresholds' not in adata.uns:

            if self.status_bar:
                self.status_bar.update('Loading ligand-receptor database...')
            df_ligrec = get_cellchat_db(self.species) 
            df_ligrec['name'] = df_ligrec['ligand'] + '-' + df_ligrec['receptor']
            
            if self.status_bar:
                self.status_bar.update('🔬 Commot: Expanding paired interactions...')
            expanded = expand_paired_interactions(df_ligrec)
            genes = set(expanded.ligand) | set(expanded.receptor)
            genes = list(genes)

            expanded = expanded[
                expanded.ligand.isin(adata.var_names) & expanded.receptor.isin(adata.var_names)]
            
            adata.X = adata.layers['normalized_count']
            
            if self.status_bar:
                self.status_bar.update('🔬 COMMOT: Computing spatial communication...')
            ct.tl.spatial_communication(adata,
                database_name='user_database', 
                df_ligrec=expanded, 
                dis_thr=radius, 
                heteromeric=False
            )
            
            expanded['rename'] = expanded['ligand'] + '-' + expanded['receptor']
            
            if self.status_bar:
                self.status_bar.update(f'Computing cluster communication for {len(expanded["rename"].unique())} pathways...')
            unique_pathways = expanded['rename'].unique()
            for idx, name in enumerate(unique_pathways):
                if self.status_bar:
                    self.status_bar.update(f'🔬 Commot: Cluster communication {idx+1}/{len(unique_pathways)}: {name[:30]}...')
                ct.tl.cluster_communication(
                    adata, 
                    database_name='user_database', 
                    pathway_name=name, 
                    clustering='cell_type',
                    random_seed=42, 
                    n_permutations=100
                )
                
            data_dict = defaultdict(dict)

            for name in expanded['rename']:
                data_dict[name]['communication_matrix'] = adata.uns[
                    f'commot_cluster-cell_type-user_database-{name}']['communication_matrix']
                data_dict[name]['communication_pvalue'] = adata.uns[
                    f'commot_cluster-cell_type-user_database-{name}']['communication_pvalue']

            with open(f'{self.outdir}/input_data/communication.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
                
            info = data_dict
            
            def get_sig_interactions(value_matrix, p_matrix, pval=0.3):
                p_matrix = np.where(p_matrix < pval, 1, 0)
                return value_matrix * p_matrix
            
            if self.status_bar:
                self.status_bar.update('Processing significant interactions...')
            interactions = {}
            for lig, rec in tqdm(zip(expanded['ligand'], expanded['receptor'])):
                name = lig + '-' + rec
                if name in info.keys():
                    value_matrix = info[name]['communication_matrix']
                    p_matrix = info[name]['communication_pvalue']
                    sig_matrix = get_sig_interactions(value_matrix, p_matrix)
                    if sig_matrix.sum().sum() > 0:
                        interactions[name] = sig_matrix
                        
            if self.status_bar:
                self.status_bar.update('Computing ligand-receptor thresholds...')
            ct_masks = {cell_type: adata.obs[self.annot] == cell_type for cell_type in adata.obs[self.annot].unique()}
            df = pd.DataFrame(index=adata.obs_names, columns=genes)
            df = df.fillna(0)
            for name in tqdm(interactions.keys(), total=len(interactions)):
                lig, rec = name.rsplit('-', 1)
                tmp = interactions[name].sum(axis=1)
                for cell_type, val in zip(interactions[name].index, tmp):
                    df.loc[ct_masks[cell_type], lig] += tmp[cell_type]
                tmp = interactions[name].sum(axis=0)
                for cell_type, val in zip(interactions[name].columns, tmp):
                    df.loc[ct_masks[cell_type], rec] += tmp[cell_type]
                    
            perc_filtered = np.where(df > 0, 1, 0).sum().sum() / (df.shape[0] * df.shape[1])      
            
            df.to_parquet(f'{self.outdir}/input_data/LRs.parquet')
            
            adata.uns['cell_thresholds'] = df.copy()
        else:
            print('Cell thresholds already computed, skipping COMMOT...')
            df = adata.uns['cell_thresholds']
        
        if self.status_bar:
            self.status_bar.update('Caching received ligands...')
        adata = init_received_ligands(
            adata, 
            radius=radius, 
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
        adata.write_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        self.status = Status.BORED

    def setup_(
            self, 
            adata: ad.AnnData, 
            overwrite=False,
            run_celloracle=True,
            run_commot=True,
            base_GRN=None
        ):
        """
        Sets up the SpaceShip environment and runs the preprocessing pipeline.
        
        This includes creating directories, processing AnnData, running CellOracle,
        and running COMMOT.

        Parameters
        ----------
        adata : ad.AnnData
            Input AnnData object.
        overwrite : bool, optional
            If True, overwrites existing output directory, by default False.
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        if os.path.exists(self.outdir) and not overwrite:
            print("Warning: output directory already exists. Will not overwrite.")
            self.status = Status.FUBAR
            return
        
        self.manager = enlighten.get_manager()
        self.status_bar = self.manager.status_bar(
            f'🚀 SpaceShip {self.name}: Initializing...',
            color='black_on_cyan',
            justify=enlighten.Justify.CENTER,
            auto_refresh=True
        )
        
        if self.status_bar:
            self.status_bar.update('🚀 SpaceShip: Creating output directories...')
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(f'{self.outdir}/betadata', exist_ok=True)
        os.makedirs(f'{self.outdir}/input_data', exist_ok=True)
        os.makedirs(f'{self.outdir}/logs', exist_ok=True)
        
        self.status = Status.RUNNING
        
        self.process_adata_(adata)

        self.setup_tf_modulators_(run_celloracle=run_celloracle, base_GRN=base_GRN)
        self.setup_lr_modulators_(run_commot=run_commot)
        self.setup_tfl_modulators_()

        if self.status_bar:
            self.status_bar.update('✅ SpaceShip: Setup complete!')
        self.status = Status.BORED
        
        return self
    
    def setup_tf_modulators_(self, run_celloracle=True, base_GRN=None):
        if run_celloracle:
            self.run_celloracle_(base_GRN=base_GRN)
        else:
            from itertools import product

            if base_GRN is None:
                base_grn = self.load_base_GRN(self.species)
            else:
                base_grn = base_GRN
            tfs = base_grn.columns
            tfs = list(set(tfs) & set(self.adata.var_names) - {'peak_id', 'gene_short_name'})
            targets = base_grn['gene_short_name'].unique()
            targets = list(set(targets) & set(self.adata.var_names))

            # this is inefficient but uses the same structure as with TF-target priors
            
            pairs = list(product(tfs, targets))
            df = pd.DataFrame(pairs, columns=['source', 'target'])
            df['coef_mean'] = 1
            df['coef_abs'] = 1
            df['p'] = 1e-5
            df['-logp'] = 5

            links_dict = {ct: df for ct in self.adata.obs[self.annot].unique()}
            self.links = links_dict
        
            with open(f'{self.outdir}/input_data/celloracle_links.pkl', 'wb') as f:
                pickle.dump(links_dict, f)

    def setup_lr_modulators_(self, run_commot=True):
        if run_commot:
            self.run_commot_()
        else:
            return
    
    def setup_tfl_modulators_(self):
        # this is species-specific, not dataset specific
        self.get_nichenet_links_()
    
    def get_nichenet_links_(self):
        if self.status_bar:
            self.status_bar.update('🔗 NicheNet: Downloading ligand-target links...')
        
        data_path = f'https://zenodo.org/records/17594271/files/ligand_target_{self.species}.parquet'
        nichenet_lt = pd.read_parquet(data_path)
        
        if self.status_bar:
            self.status_bar.update('🔗 NicheNet: Saving links...')
        nichenet_lt.to_parquet(f'{self.outdir}/input_data/tflinks.parquet')
        
        if self.status_bar:
            self.status_bar.update('✅ NicheNet: Complete')
        return nichenet_lt
        
    def spawn_worker(
        self, 
        partition='preempt', 
        clusters='gpu', 
        gres='gpu:1', 
        job_name='SpaceTravLR',
        lifespan=3, # hours
        python_path='python',
        ):
        """
        Submits a SLURM job to run the analysis.
        
        Parameters
        ----------
        partition : str, optional
            SLURM partition, by default 'preempt'.
        clusters : str, optional
            SLURM cluster, by default 'gpu'.
        gres : str, optional
            Generic Resource Scheduling (e.g. gpu:1), by default 'gpu:1'.
        job_name : str, optional
            Name of the job, by default 'SpaceTravLR'.
        lifespan : int, optional
            Wall-time in hours, by default 3.
        python_path : str, optional
            Path to python executable, by default 'python'.
        """
        
        outlog = f'{self.outdir}/logs/training_{str(time.strftime("%Y%m%d_%H%M%S"))}.log'
        
        slurm = Slurm(
            cpus_per_task=1,
            partition=partition,
            clusters=clusters,
            gres=gres,
            ignore_pbs=True,
            job_name=job_name+'_'+self.name,
            output=outlog,
            time=timedelta(hours=lifespan),
        ) 
        
        slurm.sbatch(python_path + ' launch.py')
        
    @catch_errors
    def run_spacetravlr(
        self, 
        max_epochs: int = 150, 
        learning_rate: float = 5e-3, 
        spatial_dim: int = 64, 
        batch_size: int = 512, 
        radius: int = 300, 
        contact_distance: int = 50,
        extra_modulators: list[str] = None,
        extra_lr: list[tuple[str, str]] = None,
        activation: str = 'sigmoidx2',
        scale_factor: int = 100,
        save_models: bool = False
    ):
        """
        Trains the SpaceTravLR model to learn spatial gene regulation.
        
        This method initializes and trains the SpaceTravLR neural network
        model to predict gene expression based on TF activity and spatial
        ligand-receptor interactions.

        Parameters
        ----------
        max_epochs : int, optional
            Maximum number of training epochs, by default 150.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 5e-3.
        spatial_dim : int, optional
            Dimension of the spatial embedding, by default 64.
        batch_size : int, optional
            Batch size for training, by default 512.
        radius : int, optional
            Radius for secreted signaling, by default 300.
        contact_distance : int, optional
            Distance for contact-dependent signaling, by default 50.
        """
        
        from .oracles import SpaceTravLR
        from .tools.network import RegulatoryFactory
        
        base_dir = f'{self.outdir}/betadata/'
        adata = sc.read_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        tflinks = pd.read_parquet(f'{self.outdir}/input_data/tflinks.parquet')
        links = pickle.load(open(f'{self.outdir}/input_data/celloracle_links.pkl', 'rb'))

        co_grn = RegulatoryFactory(links=links)
        
        space_travlr = SpaceTravLR(
            adata=adata,
            max_epochs=max_epochs, 
            learning_rate=learning_rate, 
            spatial_dim=spatial_dim,
            batch_size=batch_size,
            grn=co_grn,
            radius=radius,
            contact_distance=contact_distance,
            save_dir=base_dir,
            tflinks=tflinks,
            scale_factor=scale_factor,
            activation=activation,
            extra_modulators=extra_modulators,
            extra_lr=extra_lr,
            save_models=save_models
        )


        space_travlr.run()

    #@alias
    def fit(self, **kwargs): return self.run_spacetravlr(**kwargs)
    
    def load_estimators(self, n_clusters=-1, genes=None, rust_version=False):
        """
        Loads trained SpatialCellularProgramsEstimator objects from the output directory.
        
        Parameters
        ----------
        genes : list, optional
            List of genes to load models for. If None, loads all available models.
        """
        import glob
        import pickle
        from .oracles import CPU_Unpickler
        
        
        if rust_version:

            model_dir = os.path.join(self.outdir, 'betadata', 'CNN_weights')
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found at {model_dir}")


            from .models.load_cnn_npz_pytorch import load_cluster_model

            if genes is None:
                model_paths = glob.glob(os.path.join(model_dir, '*.npz'))
            else:
                model_paths = [os.path.join(model_dir, f'{gene}_cnn_weights.npz') for gene in genes]
                model_paths = [p for p in model_paths if os.path.exists(p)]

            self.estimators = {}
            for path in tqdm(model_paths, desc="Loading estimators"):
                gene = os.path.basename(path).replace('_cnn_weights.npz', '')

                if n_clusters <= 0:
                    import numpy as np
                    data = np.load(path, allow_pickle=False)
                    cluster_keys = [f for f in data.files if f.endswith('_spatial_l1_weight')]
                    n_cl = len(cluster_keys)
                    if n_cl == 0:
                        raise ValueError(f"Could not determine number of clusters from weights file: {path}")
                else:
                    n_cl = n_clusters

                self.estimators[gene] = {
                    cluster_id: load_cluster_model(path, cluster_id=cluster_id)
                    for cluster_id in range(n_cl)
                }

        else: 
            model_dir = os.path.join(self.outdir, 'betadata', 'models')
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found at {model_dir}")

            if genes is None:
                model_paths = glob.glob(os.path.join(model_dir, '*.pkl'))
            else:
                model_paths = [os.path.join(model_dir, f'{gene}.pkl') for gene in genes]
                model_paths = [p for p in model_paths if os.path.exists(p)]
                
            self.estimators = {}
            for path in tqdm(model_paths, desc="Loading estimators"):
                gene = os.path.basename(path).replace('.pkl', '')
                with open(path, 'rb') as f:
                    # Use CPU_Unpickler in case models were saved on GPU
                    self.estimators[gene] = CPU_Unpickler(f).load()
                    
        print(f"Loaded {len(self.estimators)} estimators.")

    def get_betas_on_new_data(self, new_adata, rust_version=False):
        """
        Applies loaded trained models to a new AnnData object to compute spatial betas.
        
        Parameters
        ----------
        new_adata : ad.AnnData
            The new AnnData object to compute betas for.
            Must have 'spatial' in obsm and the same cell type annotations.
        
        Returns
        -------
        dict
            A dictionary of beta DataFrames for each gene.
        """
        if not hasattr(self, 'estimators') or not self.estimators:
            print("No estimators loaded. Loading all available estimators...")
            self.load_estimators()
            
        if not self.estimators:
            raise ValueError("No estimators found to perform prediction.")

        if rust_version:

            from .models.parallel_estimators import create_spatial_features
            from .models.spatial_map import xyc2spatial_fast
            from sklearn.preprocessing import MinMaxScaler
            import tomllib
            import torch

            with open(f"{self.outdir}/betadata/spacetravlr_run_repro.toml", "rb") as f:
                config = tomllib.load(f)
            
            dummy_gene = list(self.estimators.keys())[0]
            annot = config['data']['cluster_annot']

            new_adata.obs[annot] = new_adata.obs[annot].astype(int)
            
            # Determine the full set of cluster IDs from the loaded estimators
            # to guarantee input feature dimensions match the trained model shapes
            trained_clusters = sorted(list(self.estimators[dummy_gene].keys()))
            n_clusters = len(trained_clusters)

            print('WARNING: extra LRs are not compatible yet')
            # Generate spatial maps using the full set of trained clusters
            sp_maps = xyc2spatial_fast(
                xyc = np.column_stack([new_adata.obsm['spatial'], new_adata.obs[annot]]),
                m=config['spatial']['spatial_dim'],
                n=config['spatial']['spatial_dim'],
                clusters=trained_clusters
            )

            # Generate spatial features using the full set of trained clusters
            spatial_features = create_spatial_features(
                x=new_adata.obsm['spatial'][:, 0], 
                y=new_adata.obsm['spatial'][:, 1], 
                celltypes=new_adata.obs[annot], 
                obs_index=new_adata.obs.index,
                celltypes_order=trained_clusters,
                radius=config['spatial']['radius']
            )
            # Scale spatial features as done during model training
            spatial_features = pd.DataFrame(
                MinMaxScaler().fit_transform(spatial_features.values), 
                columns=spatial_features.columns, 
                index=spatial_features.index
            )

            all_betas = {}
            for gene, estimator in tqdm(self.estimators.items(), desc="Predicting betas"):
                
                betadata_ref = pd.read_feather(f'{self.outdir}/betadata/{gene}_betadata.feather')
                betadata_ref.set_index('CellID', inplace=True)
                goi_betadata = pd.DataFrame(index=new_adata.obs.index, columns=betadata_ref.columns)
                
                # Predict betas for the cell types present in the new data
                unique_new_clusters = sorted(new_adata.obs[annot].unique().tolist())
                for cluster_id in unique_new_clusters:
                    if cluster_id not in estimator:
                        continue
                    
                    # Create boolean mask for cells belonging to this cluster
                    mask = (new_adata.obs[annot] == cluster_id).values
                    if not np.any(mask):
                        continue
                    
                    # Get the model and its device
                    model = estimator[cluster_id]
                    device = next(model.parameters()).device
                    
                    # Correctly slice sp_maps to (n_cells_in_cluster, 1, m, n) and send to device
                    cluster_idx = trained_clusters.index(cluster_id)
                    cluster_sp_maps = torch.from_numpy(
                        sp_maps[mask][:, cluster_idx:cluster_idx+1, :, :]
                    ).float().to(device)
                    
                    # Correctly slice spatial features to (n_cells_in_cluster, n_clusters) and send to device
                    spf = torch.from_numpy(
                        spatial_features.values[mask]
                    ).float().to(device)
                    
                    # Predict betas and convert back to numpy
                    cluster_betas = model.get_betas(
                        spatial_maps=cluster_sp_maps, 
                        spatial_features=spf
                    ).detach().cpu().numpy()

                    import pdb; pdb.set_trace()

                    # Assign predicted betas to the corresponding rows in goi_betadata
                    goi_betadata.loc[mask, betadata_ref.columns] = cluster_betas

                all_betas[gene] = goi_betadata

        else:

            all_betas = {}
            for gene, estimator in tqdm(self.estimators.items(), desc="Predicting betas"):
                # Ensure the estimator uses the new adata
                # We call init_data on the estimator with the new adata 
                # to compute spatial features and received ligands
                estimator.init_data(new_adata)
                all_betas[gene] = estimator.get_betas()
                
        return all_betas
    
    def setup_perturbations(self, adata, override_params=None, subsample=None, use_float16=False):
        """
        Initializes the GeneFactory for running perturbations.
        
        Parameters
        ----------
        adata : ad.AnnData
            AnnData object used for perturbation simulations.
        override_params : dict, optional
            Dictionary to override run parameters, by default None.
        subsample : int, optional
            Number of cells to subsample for faster loading, by default None.
        use_float16 : bool, optional
            Use float16 for lower memory usage, by default False.
        """
        from .gene_factory import GeneFactory
        json_path = f'{self.outdir}/betadata/run_params.json'
        assert os.path.exists(json_path), f"run_params.json not found"
        
        self.factory = GeneFactory.from_json(
            adata=adata, 
            json_path=json_path,
            override_params=override_params
        )
        
        self.factory.load_betas(subsample=subsample, float16=use_float16)

    def perturb(self, target, propagation=4, gene_expr=0, cells=None):
        """
        Performs in silico perturbation of a target gene.
        
        Simulates the effect of changing a gene's expression (knockout or
        overexpression) on the entire transcriptome, considering spatial
        signaling propagation.

        Parameters
        ----------
        target : str or list
            Target gene(s) to perturb.
        propagation : int, optional
            Number of propagation steps (hops) in the network, by default 4.
        gene_expr : float or list, optional
            Target expression level (0 for knockout), by default 0.
        cells : list, optional
            List of cell indices to apply perturbation to (None for all cells),
            by default None.
        
        Returns
        -------
        pd.DataFrame
            Simulated gene expression matrix after perturbation.
        """
        return self.factory.perturb(
            target=target,
            n_propagation=propagation,
            gene_expr=gene_expr,
            cells=cells
        )

    def is_everything_ok(self) -> bool:
        """
        Checks if all necessary output files and directories exist.
        
        Returns
        -------
        bool
            True if all checks pass.
        """
        assert os.path.isfile(f'{self.outdir}/input_data/_adata.h5ad'), "AnnData file not found"
        _adata = sc.read_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        _links = pickle.load(open(f'{self.outdir}/input_data/celloracle_links.pkl', 'rb'))
        
        assert 'imputed_count' in _adata.layers, "Imputed count layer not found"
        assert 'X_umap' in _adata.obsm, "UMAP embedding not found"
        assert 'cell_type_int' in _adata.obs.columns, "Cell type integer column not found"
        assert 'spatial' in _adata.obsm, "Spatial coordinates not found"
        
        assert os.path.isdir(self.outdir), "Output directory not found"
        assert os.path.isdir(f'{self.outdir}/betadata'), "Betadata directory not found"
        assert os.path.isdir(f'{self.outdir}/input_data'), "Input data directory not found"
        assert os.path.isfile(f'{self.outdir}/input_data/celloracle_links.pkl'), "Base links file not found"
        assert os.path.isdir(f'{self.outdir}/logs'), "Logs directory not found"
        assert os.path.isfile('launch.py'), "Launch script not found"
        
        print("We're going on a trip in our favorite rocket ship 🚀️")
        
        return True
    
    @catch_errors
    def sweep_spacetravlr(self, target_genes: list[str], wandb_project: str, wandb_name: str, training_params: dict):
        
        from .tools.network import RegulatoryFactory
        from .models.parallel_estimators import init_received_ligands
        from .models.parallel_estimators import SpatialCellularProgramsEstimator, create_spatial_features
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import r2_score
        import time
        import wandb

        wandb_logs = {}

        # initialize wandb
        # wandb.init(
        #     project=wandb_project, 
        #     name=wandb_name
        # )
        # for key, value in training_params.items():
        #     wandb.config[key] = value
        
        # load data from setup
        adata_base = sc.read_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        tflinks = pd.read_parquet(f'{self.outdir}/input_data/tflinks.parquet')
        links = pickle.load(open(f'{self.outdir}/input_data/celloracle_links.pkl', 'rb'))

        co_grn = RegulatoryFactory(links=links)

        n_hvgs = training_params['n_hvgs']
        n_cells = training_params['n_cells']

        # temporary fix because I forgot to remove uninformative genes
        adata_base.var_names_make_unique()
        adata_base.var["MT"] = adata_base.var_names.str.startswith("MT-")

        sc.pp.calculate_qc_metrics(adata_base, qc_vars=["MT"], inplace=True)
        sc.pp.filter_cells(adata_base, min_counts=50)
        adata_base = adata_base[adata_base.obs["pct_counts_MT"] < 10].copy()
        adata_base = adata_base[:, ~adata_base.var["MT"]]

        adata_base = adata_base[:, ~adata_base.var_names.str.contains('RIK')]
        adata_base = adata_base[:, ~adata_base.var_names.str.contains(r'^HB\w+-\w+$')]
        adata_base = adata_base[:, ~adata_base.var_names.str.contains('HP')]
        adata_base = adata_base[:, ~adata_base.var_names.str.startswith('RP')]
        adata_base = adata_base[:, ~adata_base.var_names.str.startswith('AA')]
        adata_base = adata_base[:, ~adata_base.var_names.str.startswith('AB')]
        adata_base = adata_base[:, ~adata_base.var_names.str.startswith('AC')]
        adata_base = adata_base[:, ~adata_base.var_names.str.startswith('GM')]
        adata_base = adata_base[:, ~adata_base.var_names.str.startswith('MIR')]
        adata_base = adata_base[:, ~adata_base.var_names.str.startswith('TTT')]
        adata_base = adata_base[:, ~adata_base.var_names.str.startswith('LINC')]
        adata_base = adata_base[:, ~adata_base.var_names.str.endswith('-AS1')]

        housekeeping_genes = pd.read_csv('/ix/djishnu/shared/djishnu_kor11/tonsil_sweep/Housekeeping_GenesHuman.csv', index_col=0, sep=';')
        housekeeping_genes = housekeeping_genes['Gene.name'].tolist()
        adata_base = adata_base[:, ~adata_base.var_names.isin(housekeeping_genes)]
        sc.pp.filter_genes(adata_base, min_cells=10)

        # Subset into train and val
        adata = adata_base.copy()
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvgs, 
            batch_key='cell_type_int', 
            flavor='seurat_v3', 
            inplace=True
        )
        train_cells = np.random.choice(adata.obs_names, size=n_cells, replace=False)
        adata_train = adata[train_cells]
        adata_val = adata[~adata.obs_names.isin(train_cells)]
        
        # make sure that the density of the validation set is the same as the training set
        if len(adata_val) < n_cells:
            extra_cells = np.random.choice(train_cells, size=n_cells - len(adata_val), replace=False)
            adata_val = sc.concat([adata_val, adata[extra_cells]], axis=0)
        if len(adata_val) > n_cells:
            adata_val = adata_val[np.random.choice(adata_val.obs_names, size=n_cells, replace=False)]

        adata_train = init_received_ligands(
            adata_train, 
            radius=training_params['radius'], 
            cell_threshes=adata_train.uns.get('cell_thresholds', None),
            scale_factor=training_params['scale_factor']
        )

        adata_val = init_received_ligands(
            adata_val, 
            radius=training_params['radius'], 
            cell_threshes=adata_val.uns.get('cell_thresholds', None),
            scale_factor=training_params['scale_factor']
        )

        def sp_feature_from_adata(adata, radius):
            spatial_features = create_spatial_features(
                adata.obsm['spatial'][:, 0], 
                adata.obsm['spatial'][:, 1], 
                adata.obs['cell_type_int'], 
                adata.obs.index,
                radius=radius
            )
            spatial_features = pd.DataFrame(
                MinMaxScaler().fit_transform(spatial_features.values), 
                columns=spatial_features.columns, 
                index=spatial_features.index
            )
            return spatial_features

        adata_val.obsm['spatial_features'] = sp_feature_from_adata(
            adata_val,
            radius=training_params['radius']
        )

        adata_train.obsm['spatial_features'] = sp_feature_from_adata(
            adata_train,
            radius=training_params['radius']
        )

        start_time = time.time()

        # initialize estimator
        for target_gene in target_genes:
            
            print(f"Starting training for {target_gene}")
            
            estimator = SpatialCellularProgramsEstimator(
                adata=adata_train,
                target_gene=target_gene,
                layer='imputed_count',
                cluster_annot='cell_type_int',
                spatial_dim=training_params['spatial_dim'],
                radius=training_params['radius'],
                contact_distance=training_params['contact_distance'],
                tf_ligand_cutoff=training_params['tf_ligand_cutoff'],
                receptor_thresh=training_params['receptor_thresh'],
                grn=co_grn,
                use_ligands=training_params['use_ligands'],
                tflinks=tflinks,
                activation=training_params['activation'],
                scale_factor=training_params['scale_factor']
            )

            estimator.fit(
                num_epochs=training_params['max_epochs'],
                learning_rate=training_params['learning_rate'],
                batch_size=training_params['batch_size'],
                use_pbar=False,
                estimator='lasso',
                vision_model='cnn',
                lasso_params=training_params['lasso_params']
            )

            for ct in adata_train.obs['cell_type_int'].unique():

                y_train, y_train_pred = estimator.predict(ct, adata_train)
                r2_train = r2_score(y_train, y_train_pred)
                
                y_val, y_val_pred = estimator.predict(ct, adata_val)
                r2_val = r2_score(y_val, y_val_pred)

                # sometimes we zero out the prediction because of bad values
                if y_train_pred.sum() <= 0:
                    continue
                else:
                    wandb_logs[f'r2_train_{target_gene}_{ct}'] = r2_train
                    wandb_logs[f'r2_val_{target_gene}_{ct}'] = r2_val

        end_time = time.time()

        wandb_logs['training_time'] = end_time - start_time
        r2_val_keys = [k for k in wandb_logs if k.startswith('r2_val_')]
        r2_train_keys = [k for k in wandb_logs if k.startswith('r2_train_')]
        wandb_logs['r2_val_mean'] = np.mean([wandb_logs[k] for k in r2_val_keys])
        wandb_logs['r2_train_mean'] = np.mean([wandb_logs[k] for k in r2_train_keys])
        wandb.log(wandb_logs)
