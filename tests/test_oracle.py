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

from spaceoracle.tools.network import DayThreeRegulatoryNetwork
from spaceoracle.oracles import *
# from spaceoracle.models.probabilistic_estimators import ProbabilisticPixelModulators
from spaceoracle.models.parallel_estimators import received_ligands

import anndata as ad

def generate_realistic_data(noise_level=0.1):
    np.random.seed(42)
    adata = ad.read_h5ad('./data/slideseq/day3_1.h5ad')
    # adata = ad.read_h5ad('/ix/djishnu/alw399/SpaceOracle/data/slideseq/day3_1.h5ad')
    grn = DayThreeRegulatoryNetwork()

    regulators = grn.get_regulators(adata, 'Cd74')[:5]

    adata = adata[:, adata.var_names.isin(regulators+['Cd74']+['Il2', 'Il2ra', 'Ccl5', 'Bmp2', 'Bmpr1a'])]

    adata = adata[adata.obs['rctd_cluster'].isin([0, 1])]
    adata = adata[:600, :]

    adata.obs['rctd_cluster'] = adata.obs['rctd_cluster'].cat.remove_unused_categories()

    adata.layers['imputed_count'] = adata.X.toarray().copy()
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

def test_oracle_initialization(mock_adata_with_true_betas):
    adata = mock_adata_with_true_betas
    oracle = BaseTravLR(adata)
    assert 'imputed_count' in oracle.adata.layers
    assert oracle.pcs is None
    assert oracle.gene2index is not None

    del adata.layers['imputed_count']
    adata = mock_adata_with_true_betas
    oracle = BaseTravLR(adata)
    assert 'imputed_count' in oracle.adata.layers
    assert oracle.pcs is not None
    assert oracle.gene2index is not None


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

def test_space_oracle_initialization(mock_adata_with_true_betas, temp_dir):
    adata = mock_adata_with_true_betas
    space_oracle = SpaceTravLR(adata, save_dir=temp_dir)
    assert space_oracle.adata is not None
    assert space_oracle.grn is not None
    assert space_oracle.queue is not None

# @pytest.mark.parametrize("estimator_class", [ProbabilisticPixelModulators])
# def test_space_oracle_run(mock_adata_with_true_betas, temp_dir, estimator_class):
#     adata = mock_adata_with_true_betas
#     with patch('spaceoracle.oracles.PixelAttention', MagicMock(return_value=estimator_class(adata, 'Cd74'))):
#         space_oracle = SpaceOracle(adata, save_dir=temp_dir, max_epochs=2, batch_size=3)
#         space_oracle.adata.uns['received_ligands'] = ProbabilisticPixelModulators.received_ligands(
#             xy=adata.obsm['spatial'], 
#             lig_df=adata.to_df()[[adata.var_names[0]]], 
#             radius=10, 
#         )
#         space_oracle.run()

#     assert len(space_oracle.queue.completed_genes) > 0
#     assert len(space_oracle.trained_genes) > 0
#     assert len(os.listdir(temp_dir)) > 0


def test_lr_radius(mock_adata_with_true_betas):
    adata = mock_adata_with_true_betas
    sim_lr = generate_simulated_lr()
    ligands = ['Ccl5', 'Il2']
    ligands_df = adata.to_df(layer='imputed_count')[ligands]

    weighted_ligands = received_ligands(
        adata.obsm['spatial'],
        ligands_df, 
        sim_lr
    )

    neighbors = get_neighbors_within_radius(adata, 100)
    for i, ligand_vals in enumerate(weighted_ligands['Ccl5']):
        raw_vals = bool(ligands_df['Ccl5'].iloc[neighbors[i]].sum())
        ligand_vals = bool(ligand_vals)

        assert raw_vals == ligand_vals, f'Failed for cell {i}'

    neighbors = get_neighbors_within_radius(adata, 30)
    for i, ligand_vals in enumerate(weighted_ligands['Il2']):
        raw_vals = bool(ligands_df['Il2'].iloc[neighbors[i]].sum())
        ligand_vals = bool(ligand_vals)

        assert raw_vals == ligand_vals, f'Failed for cell {i}'
