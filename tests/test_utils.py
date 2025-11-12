import pytest
import numpy as np
import pandas as pd
import anndata
from scipy import sparse
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from SpaceTravLR.tools.utils import is_mouse_data, gaussian_kernel_2d, scale_adata


class TestIsMouseData:
    def test_mouse_data_detection(self):
        """Test that mouse data (first letter capitalized) is correctly identified."""
        # Create mock AnnData with mouse-style gene names
        var_names = pd.Index(['Gata1', 'Pax6', 'Sox2', 'Nanog', 'Klf4'])
        mock_adata = anndata.AnnData(X=np.zeros((3, 5)), var=pd.DataFrame(index=var_names))
        
        assert is_mouse_data(mock_adata) is True
    
    def test_human_data_detection(self):
        """Test that human data (all caps) is correctly identified."""
        # Create mock AnnData with human-style gene names
        var_names = pd.Index(['GATA1', 'PAX6', 'SOX2', 'NANOG', 'KLF4'])
        mock_adata = anndata.AnnData(X=np.zeros((3, 5)), var=pd.DataFrame(index=var_names))
        
        assert is_mouse_data(mock_adata) is False
    
    def test_mixed_data(self):
        """Test behavior with mixed naming conventions."""
        # Create mock AnnData with mixed gene naming styles
        var_names = pd.Index(['GATA1', 'Pax6', 'SOX2', 'Nanog', 'KLF4'])
        mock_adata = anndata.AnnData(X=np.zeros((3, 5)), var=pd.DataFrame(index=var_names))
        
        # The function should return based on which pattern is more common
        result = is_mouse_data(mock_adata)
        # In this case, human pattern (3) > mouse pattern (2)
        assert result is False
    
    def test_edge_case_empty_data(self):
        """Test behavior with empty var_names."""
        var_names = pd.Index([])
        mock_adata = anndata.AnnData(X=np.zeros((3, 0)), var=pd.DataFrame(index=var_names))
        
        # With no genes to check, the function should default to one or the other
        # Let's assert it doesn't crash
        result = is_mouse_data(mock_adata)
        assert isinstance(result, bool)


class TestGaussianKernel2D:
    def test_basic_functionality(self):
        """Test basic functionality of gaussian_kernel_2d."""
        origin = np.array([0, 0])
        xy_array = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        radius = 2.0
        
        weights = gaussian_kernel_2d(origin, xy_array, radius)
        
        # Check shape
        assert weights.shape == (4,)
        
        # Check values
        assert weights[0] > 0.99  # Point at origin should have weight close to 1
        assert 0 < weights[1] < 1  # Points at distance 1 should have weight between 0 and 1
        assert 0 < weights[2] < 1
        assert 0 < weights[3] < 1
        
        # Check that weights decrease with distance
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]
        assert weights[0] > weights[3]
        assert weights[1] > weights[3]  # Point at [1,1] is further than point at [1,0]
        assert weights[2] > weights[3]  # Point at [1,1] is further than point at [0,1]
    
    def test_radius_effect(self):
        """Test that larger radius produces more spread out weights."""
        origin = np.array([0, 0])
        xy_array = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        
        # Small radius
        weights_small = gaussian_kernel_2d(origin, xy_array, radius=1.0)
        
        # Large radius
        weights_large = gaussian_kernel_2d(origin, xy_array, radius=3.0)
        
        # Points away from origin should have higher weights with larger radius
        assert np.all(weights_large[1:] > weights_small[1:])
    
    def test_epsilon_effect(self):
        """Test that epsilon parameter affects the spread of the kernel."""
        origin = np.array([0, 0])
        xy_array = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        radius = 2.0
        
        # Default epsilon
        weights_default = gaussian_kernel_2d(origin, xy_array, radius)
        
        # Smaller epsilon (sharper falloff)
        weights_small_eps = gaussian_kernel_2d(origin, xy_array, radius, eps=0.0001)
        
        # Larger epsilon (wider spread)
        weights_large_eps = gaussian_kernel_2d(origin, xy_array, radius, eps=0.01)
        
        # Check that smaller epsilon leads to sharper falloff
        assert np.all(weights_small_eps[1:] < weights_default[1:])
        
        # Check that larger epsilon leads to wider spread
        assert np.all(weights_large_eps[1:] > weights_default[1:])

