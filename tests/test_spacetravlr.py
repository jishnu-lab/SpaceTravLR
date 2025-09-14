import pytest
import numpy as np
import pandas as pd 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import scanpy as sc
import anndata as ad


def quick_normalize(adata):
    adata.layers['raw_count'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['normalized_count'] = adata.X.copy()
    return adata




def test_space_oracle_initialization():
    from spaceoracle.SpaceTravLR import SpaceShip
    import pickle
    
    adata = quick_normalize(sc.read_h5ad('./data/small_melanoma.h5ad'))
    
    ss = SpaceShip(adata)
    
    ss.run_celloracle()
    # ss.run_commot()
    ss.run_spacetravlr()
    
    assert True
    
    
if __name__ == '__main__':
    test_space_oracle_initialization()

