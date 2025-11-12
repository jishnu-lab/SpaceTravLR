import pytest
import numpy as np
import pandas as pd 
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from scipy.spatial import KDTree

from SpaceTravLR.tools.network import DayThreeRegulatoryNetwork
from SpaceTravLR.oracles import *
# from SpaceTravLR.models.probabilistic_estimators import ProbabilisticPixelModulators
from SpaceTravLR.models.parallel_estimators import received_ligands
from SpaceTravLR.gene_factory import GeneFactory

import anndata as ad

class MockRegulatoryFactory:
    """Mock RegulatoryFactory for testing purposes"""
    
    def __init__(self, colinks_path=None, links=None, annot='rctd_cluster'):
        self.colinks_path = colinks_path
        self.annot = annot
        
        # Create mock links data
        if links is None:
            self.links = self._create_mock_links()
        else:
            self.links = links
            
        # Mock cluster labels
        self.cluster_labels = {0: 'cluster_0', 1: 'cluster_1'}
    
    def _create_mock_links(self):
        """Create mock CellOracle-style links data"""
        # Mock data for cluster 0
        cluster_0_data = pd.DataFrame({
            'source': ['Foxp3', 'Stat3', 'Nfkb1', 'Jun', 'Fos'],
            'target': ['Cd74', 'Cd74', 'Cd74', 'Il2', 'Il2'],
            'coef_mean': [0.5, 0.3, 0.2, 0.4, 0.6],
            'p': [0.01, 0.02, 0.03, 0.01, 0.005]
        })
        
        # Mock data for cluster 1
        cluster_1_data = pd.DataFrame({
            'source': ['Tbx21', 'Gata3', 'Rorc', 'Bcl6', 'Prdm1'],
            'target': ['Cd74', 'Cd74', 'Il2ra', 'Il2ra', 'Ccl5'],
            'coef_mean': [0.4, 0.7, 0.3, 0.5, 0.8],
            'p': [0.02, 0.001, 0.04, 0.01, 0.003]
        })
        
        return {0: cluster_0_data, 1: cluster_1_data}
    
    def get_regulators(self, adata, target_gene, alpha=0.05):
        """Mock implementation of get_regulators"""
        regulators_with_pvalues = self.get_regulators_with_pvalues(adata, target_gene, alpha)
        if regulators_with_pvalues.empty:
            return []
        
        grouped_regulators = regulators_with_pvalues.groupby('source').mean()
        filtered_regulators = grouped_regulators[grouped_regulators.index.isin(adata.var_names)]
        
        return filtered_regulators.index.tolist()
    
    def get_regulators_with_pvalues(self, adata, target_gene, alpha=0.05):
        """Mock implementation of get_regulators_with_pvalues"""
        co_links = pd.concat([
            link_data.query(f'target == "{target_gene}" and p < {alpha}')[['source', 'coef_mean']]
            for link_data in self.links.values()
        ], axis=0).reset_index(drop=True)
        
        if co_links.empty:
            return pd.DataFrame(columns=['source', 'coef_mean'])
            
        return co_links.query(f'source.isin({str(list(adata.var_names))})').reset_index(drop=True)
    
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        """Mock implementation of get_cluster_regulators"""
        import torch
        
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()
        
        for label in adata_clusters:
            if label in self.links:
                grn_df = self.links[label]
                grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
                tfs = list(grn_df.source)
                
                regulator_dict[label] = tfs
                all_regulators.update(tfs)
        
        all_regulators = all_regulators & set(adata.var_names)
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}
        
        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)
            mask[[0] + indices] = 1
            regulator_masks[label] = mask
        
        self.regulator_dict = regulator_masks
        
        return all_regulators


def generate_realistic_data(noise_level=0.1):
    np.random.seed(42)
    adata = ad.read_h5ad('./data/snrna_germinal_center.h5ad')
    grn = MockRegulatoryFactory(
    )

    regulators = grn.get_regulators(adata, 'Cd74')[:5]

    adata = adata[:, adata.var_names.isin(regulators+['Cd74']+['Il2', 'Il2ra', 'Ccl5', 'Bmp2', 'Bmpr1a'])]

    adata = adata[adata.obs['cell_type_int'].isin([0, 1])]
    adata = adata[:600, :]

    adata.obs['cell_type_int'] = adata.obs['cell_type_int'].cat.remove_unused_categories()

    adata.layers['imputed_count'] = adata.X.copy()
    adata.layers['normalized_count'] = adata.layers['imputed_count'].copy()

    return adata

def generate_simulated_lr():
    sim_lr = pd.DataFrame({
        'ligand': {0: 'Tgfb1', 636: 'Ccl5', 647: 'Ccl5', 675: 'Ccl5', 719: 'Il2'},
        'receptor': {0: 'Tgfbr2',
            636: 'Ccr3',
            647: 'Ccr4',
            675: 'Ackr2',
            719: 'Il2rg'},
        'pathway': {0: 'TGFb', 636: 'CCL', 647: 'CCL', 675: 'CCL', 719: 'IL2'},
        'signaling': {0: 'Secreted Signaling',
            636: 'Secreted Signaling',
            647: 'Secreted Signaling',
            675: 'ECM-Receptor',
            719: 'Cell-Cell Contact'},
        'radius': {0: 100, 636: 100, 647: 100, 675: 30, 719: 30},
        'pairs': {0: 'Tgfb1$Tgfbr2',
            636: 'Ccl5$Ccr3',
            647: 'Ccl5$Ccr4',
            675: 'Ccl5$Ackr2',
            719: 'Il2$Il2rg'}
    })
    return sim_lr

def get_neighbors_within_radius(adata, radius):
    coords = adata.obsm['spatial']
    tree = KDTree(coords)
    neighbors = tree.query_ball_tree(tree, radius)
    return neighbors


@pytest.fixture
def mock_adata_with_true_betas():
    return generate_realistic_data()

@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# def test_oracle_initialization(mock_adata_with_true_betas):
#     adata = mock_adata_with_true_betas
#     oracle = BaseTravLR(adata)
#     assert 'imputed_count' in oracle.adata.layers
#     assert oracle.pcs is None
#     assert oracle.gene2index is not None

#     del adata.layers['imputed_count']
#     adata = mock_adata_with_true_betas
#     oracle = BaseTravLR(adata)
#     assert 'imputed_count' in oracle.adata.layers
#     assert oracle.pcs is not None
#     assert oracle.gene2index is not None


def test_oracle_queue_initialization(temp_dir, mock_adata_with_true_betas):
    adata = mock_adata_with_true_betas
    queue = OracleQueue(temp_dir, adata.var_names.tolist())
    assert queue.model_dir == temp_dir
    assert len(queue.all_genes) == adata.n_vars
    assert len(queue.orphans) == 0

def test_oracle_queue_operations(temp_dir):
    genes = ['gene1', 'gene2', 'gene3']
    queue = OracleQueue(temp_dir, genes)

    # Test remaining_genes
    assert set(queue.remaining_genes) == set(genes)

    # Test create_lock and delete_lock
    queue.create_lock('gene1')
    assert 'gene1.lock' in os.listdir(temp_dir)
    assert set(queue.remaining_genes) == {'gene2', 'gene3'}

    queue.delete_lock('gene1')
    assert 'gene1.lock' not in os.listdir(temp_dir)
    assert set(queue.remaining_genes) == set(genes)

    # Test add_orphan
    queue.add_orphan('gene2')
    assert queue.orphans == ['gene2']
    assert set(queue.remaining_genes) == set(genes)-{'gene2'}

    # Test completed_genes
    with open(os.path.join(temp_dir, 'gene1_betadata.parquet'), 'w') as f:
        f.write('dummy')
    assert queue.completed_genes == ['gene1']
    assert set(queue.remaining_genes) == {'gene3'}

def test_genome_screen(mock_adata_with_true_betas, temp_dir):
    adata = mock_adata_with_true_betas
    
    # Initialize GeneFactory
    factory = GeneFactory(
        adata, 
        models_dir=temp_dir,
        annot='rctd_cluster',
        radius=100,
        contact_distance=30
    )
    
    # Create a few mock beta files to simulate completed genes
    test_genes = ['Cd74', 'Il2', 'Ccl5']
    for gene in test_genes:
        # Create empty beta files
        with open(os.path.join(temp_dir, f'{gene}_betadata.parquet'), 'w') as f:
            f.write('dummy')
    
    # Mock the perturb method to avoid actual computation
    with patch.object(GeneFactory, 'perturb') as mock_perturb:
        mock_perturb.return_value = pd.DataFrame(
            np.random.rand(adata.n_obs, adata.n_vars),
            index=adata.obs_names,
            columns=adata.var_names
        )
        
        # Mock possible_targets property
        with patch('SpaceTravLR.gene_factory.GeneFactory.possible_targets', new=test_genes):
            # Create a subdirectory for genome screen results
            screen_dir = os.path.join(temp_dir, 'screen_results')
            os.makedirs(screen_dir, exist_ok=True)
            
            # Run genome screen
            factory.genome_screen(screen_dir, n_propagation=2)
            
            # Verify perturb was called for each gene
            assert mock_perturb.call_count == len(test_genes)
            
            # Verify output files were created
            for gene in test_genes:
                output_file = os.path.join(screen_dir, f'{gene}_2n_0x.parquet')
                assert os.path.exists(output_file)
                
            # Verify perturb was called with correct parameters
            for gene in test_genes:
                mock_perturb.assert_any_call(
                    target=gene,
                    n_propagation=2,
                    gene_expr=0,
                    cells=None,
                    delta_dir=None
                )
