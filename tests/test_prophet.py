#%%
import pytest
import numpy as np
import pandas as pd 
import os
import sys
import anndata as ad
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from spaceoracle.prophets import *
from spaceoracle.models.parallel_estimators import received_ligands
from spaceoracle.tools.network import get_cellchat_db

@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# y = B(t1) + B(r1*l1) + B(r1*l2) + B(t1*l1) 
# dy/dt1 = B + B(l1)
# dy/dr1 = B(l1) + B(l2)
# dy/dl1 = B(r1) + B(t1)
# dy/dl2 = B(r1)

# Celltype A has r1*l1 but not t1*l1
# Celltype B has t1*l1 but not r1*l1
# Celltype C has both r1*l1 and t1*l1
# Celltype D has neither

# Case 1: check dy/dl1 is correct value depending on celltype (dy/dl1)
# Case 2: check w_ligands and w_tfligands are correct         (dy/dr1, dy/dt1)

target_gene = 'y'
modulator_genes = ['t1', 'r1', 'r2', 'l1', 'l2']
modulator_terms = ['beta0', 'beta_t1', 'beta_l1$r1', 'beta_l2$r1', 'beta_l1#t1']
cells = ['A', 'B', 'C', 'D']

def generate_adata():
    adata = ad.AnnData(
        X = np.ones((4, 6)),
        obs = pd.DataFrame({
            'cell_type': cells
        }, index=cells),
        var=pd.DataFrame({
            'gene': modulator_genes + [target_gene]
        }, index=modulator_genes + [target_gene])
    )
    adata.layers['normalized_count'] = adata.X
    adata.layers['imputed_count'] = adata.X
    adata.obsm['spatial'] = np.random.rand(4, 2)
    return adata

def create_betadatas(adata, temp_dir):
    betadata= pd.DataFrame(
        data = np.random.rand(len(cells), len(modulator_terms)),
        index=adata.obs_names,
        columns=modulator_terms,
    )

    betadata.loc['A', 'beta_l1#t1'] = 0
    betadata.loc['A', 'beta_t1'] = 0
    betadata.loc['B', 'beta_l1$r1'] = 0
    betadata.loc['D', 'beta_l1#t1'] = 0 

    betadata.to_parquet(os.path.join(temp_dir, 'y_betadata.parquet')) 
    return betadata

def create_lr():
    return pd.DataFrame({
        'ligand': ['l1', 'l2'],
        'receptor': ['r1' ,'r2'],
        'pathway': ['x', 'x'],
        'signaling': ['Secreted Signaling', 'Secreted Signaling'],
        'radius': [200, 200],
    })

def create_cell_threshes():
    return pd.DataFrame(
        data = [
            [1, 1, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
        ],
        index=['A', 'B', 'C', 'D'],
        columns=['r1', 'l1', 'l2'],
    )

@pytest.fixture
def lr_info():
    return create_lr()

@pytest.fixture
def cell_threshes():
    return create_cell_threshes()

@pytest.fixture
def snake(temp_dir, cell_threshes, lr_info):
    adata = generate_adata()
    adata.uns['cell_thresholds'] = cell_threshes

    betadata = create_betadatas(adata, temp_dir)

    snake = Prophet(adata, models_dir=temp_dir)
    snake.lr = lr_info
    snake.compute_betas()

    return snake

@pytest.fixture
def gex_df(snake):
    return snake.adata.to_df('imputed_count')

def test_case_1(snake, gex_df):
    
    snake.perturb(target='l1', n_propagation=2)

    dl1_df = pd.DataFrame(
        snake.adata.layers['l1_2n_0x'], 
        index=snake.adata.obs_names, 
        columns=snake.adata.var_names
    )

    assert gex_df.loc['D', 'y'] == dl1_df.loc['D', 'y'], f'Case 1 failed: dy/dl1 failed for celltype D'

def test_case_2(snake, gex_df):

    snake.perturb(target='r1', n_propagation=2)

    dr1_df = pd.DataFrame(
        snake.adata.layers['r1_2n_0x'], 
        index=snake.adata.obs_names, 
        columns=snake.adata.var_names
    )
    assert gex_df.loc['B', 'y'] == dr1_df.loc['B', 'y'], f'Case 2 failed: dy/dr1 failed for celltype B'
    assert gex_df.loc['D', 'y'] == dr1_df.loc['D', 'y'], f'Case 2 failed: dy/dr1 failed for celltype D'

def test_case_3(snake, gex_df):

    snake.perturb(target='t1', n_propagation=2)

    dt1_df = pd.DataFrame(
        snake.adata.layers['t1_2n_0x'], 
        index=snake.adata.obs_names, 
        columns=snake.adata.var_names
    )
    assert gex_df.loc['A', 'y'] == dt1_df.loc['A', 'y'], f'Case 3 failed: dy/dt1 failed for celltype A'




