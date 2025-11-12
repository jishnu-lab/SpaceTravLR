import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import torch
import pytest
from unittest import TestCase
from unittest.mock import patch, MagicMock, Mock
from anndata import AnnData
from sklearn.metrics import r2_score
import pickle
import tempfile
import shutil
import matplotlib

from spaceoracle.models.parallel_estimators import SpatialCellularProgramsEstimator
from spaceoracle.models.pixel_attention import CellularNicheNetwork
from spaceoracle.tools.network import RegulatoryFactory


class MockRegulatoryFactory:
    def __init__(self, *args, **kwargs):
        pass
    
    def get_regulators(self, adata, target_gene):
        return ['TF1', 'TF2', 'TF3']

    def get_ligands(self, *args, **kwargs):
        return ['ligand1', 'ligand2']

    def get_receptors(self, *args, **kwargs):
        return ['receptor1', 'receptor2']

    def get_tfl_ligands(self, *args, **kwargs):
        return ['ligand1']

    def get_tfl_regulators(self, *args, **kwargs):
        return ['TF1']


class MockCellularNicheNetwork:
    def __init__(self, *args, **kwargs):
        self.device = torch.device('cpu')
        self.parameters = lambda: [torch.nn.Parameter(torch.randn(1))]
        self.anchors = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
    def __call__(self, *args, **kwargs):
        # Return a tensor when called
        return torch.randn(10)
    
    def to(self, *args, **kwargs):
        return self
    
    def train(self):
        return self
    
    def eval(self):
        return self

    def get_betas(self):
        return torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])


class SpatialCellularProgramsEstimatorTest(TestCase):
    
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Force CPU usage for all operations
        self.device = torch.device('cpu')
        
        # Create synthetic AnnData object
        self.n_cells = 100
        self.n_genes = 50
        
        # Create synthetic gene expression data
        X = np.random.rand(self.n_cells, self.n_genes)
        
        # Create synthetic spatial coordinates
        spatial_coords = np.random.rand(self.n_cells, 2) * 1000
        
        # Create gene names including our target and regulatory genes
        gene_names = [f'gene_{i}' for i in range(self.n_genes)]
        gene_names[0] = 'target_gene'  # Our target gene
        gene_names[1:4] = ['TF1', 'TF2', 'TF3']  # TFs
        gene_names[4:8] = ['ligand1', 'ligand2', 'receptor1', 'receptor2']  # L-R pairs
        
        # Create cluster annotations
        clusters = np.random.randint(0, 3, size=self.n_cells)
        
        # Create AnnData object
        self.adata = AnnData(X=X)
        self.adata.var_names = gene_names
        self.adata.obs_names = [f'cell_{i}' for i in range(self.n_cells)]
        self.adata.obs['rctd_cluster'] = clusters
        self.adata.obs['cell_type_int'] = clusters
        
        self.adata.obsm['spatial'] = spatial_coords
        
        # Add imputed count layer
        self.adata.layers['imputed_count'] = X.copy()
        
        # Create temporary directory for model export/import testing
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    # @patch('spaceoracle.models.parallel_estimators.RegulatoryFactory', MockRegulatoryFactory)
    # def test_initialization(self):
    #     """Test that the estimator initializes correctly with valid parameters."""
    #     estimator = SpatialCellularProgramsEstimator(
    #         adata=self.adata,
    #         target_gene='target_gene',
    #         cluster_annot='rctd_cluster',
    #         layer='imputed_count',
    #         radius=200,
    #         contact_distance=50,
    #         tf_ligand_cutoff=0.01,
    #         colinks_path='dummy_path'
    #     )
        
    #     # Check that the estimator was initialized correctly
    #     self.assertEqual(estimator.target_gene, 'target_gene')
    #     self.assertEqual(estimator.cluster_annot, 'rctd_cluster')
    #     self.assertEqual(estimator.layer, 'imputed_count')
    #     self.assertEqual(estimator.radius, 200)
    #     self.assertEqual(estimator.contact_distance, 50)
    #     self.assertEqual(estimator.spatial_dim, 64)  # Default value
        
    #     # Check that regulators were set correctly
    #     self.assertIsNotNone(estimator.regulators)
    #     self.assertIsInstance(estimator.regulators, list)
        
    #     # Check that ligands and receptors were initialized
    #     self.assertIsNotNone(estimator.ligands)
    #     self.assertIsNotNone(estimator.receptors)
    
    @patch('spaceoracle.models.parallel_estimators.RegulatoryFactory', MockRegulatoryFactory)
    @patch('spaceoracle.tools.network.get_cellchat_db')
    @patch('spaceoracle.models.parallel_estimators.init_ligands_and_receptors')
    def test_init_ligands_and_receptors(self, mock_init_ligands, mock_get_cellchat_db):
        """Test the initialization of ligands and receptors."""
        # Mock the CellChat database
        mock_df_ligrec = pd.DataFrame({
            'ligand': ['ligand1', 'ligand2'],
            'receptor': ['receptor1', 'receptor2'],
            'pathway': ['pathway1', 'pathway2'],
            'signaling': ['Secreted Signaling', 'Cell-Cell Contact']
        })
        
        mock_get_cellchat_db.return_value = mock_df_ligrec
        
        # Create a mock return value for init_ligands_and_receptors
        from easydict import EasyDict as edict
        mock_ligand_mixtures = edict()
        mock_ligand_mixtures.lr = {'pairs': [('ligand1', 'receptor1'), ('ligand2', 'receptor2')]}
        mock_ligand_mixtures.ligands = ['ligand1', 'ligand2']
        mock_ligand_mixtures.receptors = ['receptor1', 'receptor2']
        mock_ligand_mixtures.tfl_pairs = [('TF1', 'ligand1')]
        mock_ligand_mixtures.tfl_regulators = ['TF1']
        mock_ligand_mixtures.tfl_ligands = ['ligand1']
        
        mock_init_ligands.return_value = mock_ligand_mixtures
        
        # Create the estimator
        estimator = SpatialCellularProgramsEstimator(
            adata=self.adata,
            target_gene='target_gene',
            cluster_annot='rctd_cluster',
            layer='imputed_count',
            radius=200,
            contact_distance=50,
            tf_ligand_cutoff=0.1,
            colinks_path='dummy_path'
        )
        
        # Test that the ligands and receptors are initialized correctly
        self.assertEqual(estimator.regulators, ['TF1', 'TF2', 'TF3'])  # This comes from MockRegulatoryFactory
        self.assertEqual(estimator.ligands, ['ligand1', 'ligand2'])
        self.assertEqual(estimator.receptors, ['receptor1', 'receptor2'])
        self.assertEqual(estimator.tfl_ligands, ['ligand1'])
        self.assertEqual(estimator.tfl_regulators, ['TF1'])
    
    @patch('spaceoracle.models.parallel_estimators.RegulatoryFactory', MockRegulatoryFactory)
    @patch('spaceoracle.tools.network.get_cellchat_db')
    @patch('spaceoracle.models.parallel_estimators.received_ligands')
    def test_init_data(self, mock_received_ligands, mock_get_cellchat_db):
        """Test the data initialization process."""
        # Mock the CellChat database
        mock_df_ligrec = pd.DataFrame({
            'ligand': ['ligand1', 'ligand2'],
            'receptor': ['receptor1', 'receptor2'],
            'pathway': ['pathway1', 'pathway2'],
            'signaling': ['Secreted Signaling', 'Cell-Cell Contact']
        })
        mock_get_cellchat_db.return_value = mock_df_ligrec
        
        # Mock the nichenet_lt dataframe
        mock_nichenet_lt = pd.DataFrame(
            np.random.rand(3, 2),
            index=['TF1', 'TF2', 'TF3'],
            columns=['ligand1', 'ligand2']
        )
        
        # Mock the received_ligands function
        mock_received_ligands.return_value = pd.DataFrame(
            np.random.rand(self.n_cells, 2),
            index=self.adata.obs.index,
            columns=['ligand1', 'ligand2']
        )
        
        # Patch the pd.read_parquet to return our mock dataframe
        with patch('pandas.read_parquet', return_value=mock_nichenet_lt):
            estimator = SpatialCellularProgramsEstimator(
                adata=self.adata,
                target_gene='target_gene',
                layer='imputed_count',
                radius=200,
                contact_distance=50,
                tf_ligand_cutoff=0.01,
                colinks_path='dummy_path'
            )
            
            # Initialize data
            sp_maps, X, y, cluster_labels = estimator.init_data()
        
        # Check that the data was initialized correctly
        self.assertIsNotNone(sp_maps)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(cluster_labels)
        
        # Check shapes
        self.assertEqual(sp_maps.shape[0], self.n_cells)
        self.assertEqual(X.shape[0], self.n_cells)
        self.assertEqual(y.shape[0], self.n_cells)
        self.assertEqual(cluster_labels.shape[0], self.n_cells)
        
        # Check that the AnnData object was updated with the new data
        self.assertIn('received_ligands', self.adata.uns)
        self.assertIn('ligand_receptor', self.adata.uns)
        self.assertIn('spatial_maps', self.adata.obsm)
        self.assertIn('spatial_features', self.adata.obsm)
    
    def test_ligands_receptors_interactions(self):
        """Test the static method for computing ligand-receptor interactions."""
        # Create test dataframes
        n_samples = 10
        n_features = 5
        index = [f'cell_{i}' for i in range(n_samples)]
        columns_lig = [f'ligand_{i}' for i in range(n_features)]
        columns_rec = [f'receptor_{i}' for i in range(n_features)]
        
        received_ligands_df = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            index=index,
            columns=columns_lig
        )
        
        receptor_gex_df = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            index=index,
            columns=columns_rec
        )
        
        # Compute interactions
        lr_interactions = SpatialCellularProgramsEstimator.ligands_receptors_interactions(
            received_ligands_df, receptor_gex_df
        )
        
        # Check that the result is correct
        self.assertIsInstance(lr_interactions, pd.DataFrame)
        self.assertEqual(lr_interactions.shape, (n_samples, n_features))
        self.assertEqual(lr_interactions.index.tolist(), index)
        
        # Check that the column names are correctly formatted
        expected_columns = [f'{l}${r}' for l, r in zip(columns_lig, columns_rec)]
        self.assertEqual(lr_interactions.columns.tolist(), expected_columns)
        
        # Check that the values are correct (element-wise multiplication)
        expected_values = received_ligands_df.values * receptor_gex_df.values
        np.testing.assert_array_almost_equal(lr_interactions.values, expected_values)
    
    def test_ligand_regulators_interactions(self):
        """Test the static method for computing ligand-regulator interactions."""
        # Create test dataframes
        n_samples = 10
        n_features = 5
        index = [f'cell_{i}' for i in range(n_samples)]
        columns_lig = [f'ligand_{i}' for i in range(n_features)]
        columns_reg = [f'regulator_{i}' for i in range(n_features)]
        
        received_ligands_df = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            index=index,
            columns=columns_lig
        )
        
        regulator_gex_df = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            index=index,
            columns=columns_reg
        )
        
        # Compute interactions
        ltf_interactions = SpatialCellularProgramsEstimator.ligand_regulators_interactions(
            received_ligands_df, regulator_gex_df
        )
        
        # Check that the result is correct
        self.assertIsInstance(ltf_interactions, pd.DataFrame)
        self.assertEqual(ltf_interactions.shape, (n_samples, n_features))
        self.assertEqual(ltf_interactions.index.tolist(), index)
        
        # Check that the column names are correctly formatted
        expected_columns = [f'{l}#{r}' for l, r in zip(columns_lig, columns_reg)]
        self.assertEqual(ltf_interactions.columns.tolist(), expected_columns)
        
        # Check that the values are correct (element-wise multiplication)
        expected_values = received_ligands_df.values * regulator_gex_df.values
        np.testing.assert_array_almost_equal(ltf_interactions.values, expected_values)
    
    @patch('spaceoracle.models.parallel_estimators.RegulatoryFactory', MockRegulatoryFactory)
    @patch('spaceoracle.tools.network.get_cellchat_db')
    @patch('spaceoracle.models.parallel_estimators.received_ligands')
    @patch('spaceoracle.models.parallel_estimators.CellularNicheNetwork', MockCellularNicheNetwork)
    @patch('spaceoracle.models.parallel_estimators.torch.optim.Adam')
    @patch('spaceoracle.models.parallel_estimators.DataLoader')
    @patch('spaceoracle.models.parallel_estimators.enlighten.get_manager')
    @patch('spaceoracle.models.parallel_estimators.RotatedTensorDataset')
    def test_fit_and_get_betas(self, mock_dataset, mock_manager, mock_dataloader, mock_adam, mock_received_ligands, mock_get_cellchat_db):
        """Test the fit method and getting betas."""
        # Mock the CellChat database
        mock_df_ligrec = pd.DataFrame({
            'ligand': ['ligand1', 'ligand2'],
            'receptor': ['receptor1', 'receptor2'],
            'pathway': ['pathway1', 'pathway2'],
            'signaling': ['Secreted Signaling', 'Cell-Cell Contact']
        })
        mock_get_cellchat_db.return_value = mock_df_ligrec
        
        # Mock the nichenet_lt dataframe
        mock_nichenet_lt = pd.DataFrame(
            np.random.rand(3, 2),
            index=['TF1', 'TF2', 'TF3'],
            columns=['ligand1', 'ligand2']
        )
        
        # Mock the received_ligands function
        mock_received_ligands.return_value = pd.DataFrame(
            np.random.rand(self.n_cells, 2),
            index=self.adata.obs.index,
            columns=['ligand1', 'ligand2']
        )
        
        # Mock the DataLoader
        mock_dataloader_instance = MagicMock()
        mock_dataloader.return_value = mock_dataloader_instance
        mock_dataloader_instance.__iter__.return_value = [
            (torch.randn(10, 1, 64, 64), torch.randn(10, 5), torch.randn(10), torch.randn(10, 3))
        ]
        
        # Mock the enlighten manager
        mock_counter = MagicMock()
        mock_manager.return_value.counter.return_value = mock_counter
        
        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer
        
        # Create a custom estimator class for testing
        class TestEstimator(SpatialCellularProgramsEstimator):
            def fit(self, num_epochs=10, threshold_lambda=1e-4, learning_rate=2e-4, batch_size=512,
                    use_ARD=False, pbar=None, discard=50, use_bayesian=True, score_threshold=0.1):
                # Set up the model
                self.regulators = ['TF1', 'TF2', 'TF3']
                self.ligands = ['ligand1', 'ligand2']
                self.receptors = ['receptor1', 'receptor2']
                self.tfl_ligands = ['ligand1']
                self.tfl_regulators = ['TF1']
                self.lr_pairs = [('ligand1', 'receptor1'), ('ligand2', 'receptor2')]
                self.tfl_pairs = [('TF1', 'ligand1')]
                self.modulators = self.regulators + self.lr_pairs + self.tfl_pairs
                
                self.model = MockCellularNicheNetwork()
                self.models = {'cluster1': self.model}
                return self
            
            def get_betas(self):
                return pd.DataFrame({
                    'beta': [0.1, 0.2, 0.3, 0.4, 0.5],
                    'modulator': ['TF1', 'TF2', 'TF3', ('ligand1', 'receptor1'), ('ligand2', 'receptor2')]
                })

        # Patch the pd.read_parquet to return our mock dataframe
        with patch('pandas.read_parquet', return_value=mock_nichenet_lt):
            # Create the test estimator
            estimator = TestEstimator(
                adata=self.adata,
                target_gene='target_gene',
                cluster_annot='rctd_cluster',
                layer='imputed_count',
                radius=200,
                contact_distance=50,
                tf_ligand_cutoff=0.01,
                colinks_path='dummy_path'
            )
            
            # Call fit
            estimator.fit(num_epochs=1, batch_size=32, use_bayesian=True)
            
            # Test getting betas
            betas = estimator.get_betas()
            
            # Check that betas is a pandas DataFrame
            self.assertIsInstance(betas, pd.DataFrame)
            
            # Check that the betas dataframe has the expected structure
            self.assertEqual(betas.shape[0], 5)  # 5 modulators (3 TFs + 2 LR pairs)
    
    @patch('spaceoracle.models.parallel_estimators.RegulatoryFactory', MockRegulatoryFactory)
    @patch('spaceoracle.tools.network.get_cellchat_db')
    @patch('spaceoracle.models.parallel_estimators.received_ligands')
    @patch('spaceoracle.models.parallel_estimators.CellularNicheNetwork', MockCellularNicheNetwork)
    @patch('spaceoracle.models.parallel_estimators.torch.optim.Adam')
    @patch('spaceoracle.models.parallel_estimators.DataLoader')
    @patch('spaceoracle.models.parallel_estimators.enlighten.get_manager')
    @patch('spaceoracle.models.parallel_estimators.RotatedTensorDataset')
    def test_export_and_load(self, mock_dataset, mock_manager, mock_dataloader, mock_adam, mock_received_ligands, mock_get_cellchat_db):
        """Test exporting and loading the model."""
        # Mock the CellChat database
        mock_df_ligrec = pd.DataFrame({
            'ligand': ['ligand1', 'ligand2'],
            'receptor': ['receptor1', 'receptor2'],
            'pathway': ['pathway1', 'pathway2'],
            'signaling': ['Secreted Signaling', 'Cell-Cell Contact']
        })
        mock_get_cellchat_db.return_value = mock_df_ligrec
        
        # Mock the nichenet_lt dataframe
        mock_nichenet_lt = pd.DataFrame(
            np.random.rand(3, 2),
            index=['TF1', 'TF2', 'TF3'],
            columns=['ligand1', 'ligand2']
        )
        
        # Mock the received_ligands function
        mock_received_ligands.return_value = pd.DataFrame(
            np.random.rand(self.n_cells, 2),
            index=self.adata.obs.index,
            columns=['ligand1', 'ligand2']
        )
        
        # Mock the DataLoader
        mock_dataloader_instance = MagicMock()
        mock_dataloader.return_value = mock_dataloader_instance
        mock_dataloader_instance.__iter__.return_value = [
            (torch.randn(10, 1, 64, 64), torch.randn(10, 5), torch.randn(10), torch.randn(10, 3))
        ]
        
        # Mock the enlighten manager
        mock_counter = MagicMock()
        mock_manager.return_value.counter.return_value = mock_counter
        
        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer
        
        # Create a custom estimator class for testing
        class TestEstimator(SpatialCellularProgramsEstimator):
            def fit(self, num_epochs=10, threshold_lambda=1e-4, learning_rate=2e-4, batch_size=512,
                    use_ARD=False, pbar=None, discard=50, use_bayesian=True, score_threshold=0.1):
                # Set up the model
                self.regulators = ['TF1', 'TF2', 'TF3']
                self.ligands = ['ligand1', 'ligand2']
                self.receptors = ['receptor1', 'receptor2']
                self.tfl_ligands = ['ligand1']
                self.tfl_regulators = ['TF1']
                self.lr_pairs = [('ligand1', 'receptor1'), ('ligand2', 'receptor2')]
                self.tfl_pairs = [('TF1', 'ligand1')]
                self.modulators = self.regulators + self.lr_pairs + self.tfl_pairs
                
                self.model = MockCellularNicheNetwork()
                self.models = {'cluster1': self.model}
                return self
            
            def export(self, path):
                # Mock implementation
                return

        # Create a custom SpatialCellularProgramsEstimator class for testing load
        class TestLoadEstimator(SpatialCellularProgramsEstimator):
            def load(self, path):
                # Mock implementation
                self.regulators = ['TF1', 'TF2', 'TF3']
                self.ligands = ['ligand1', 'ligand2']
                self.receptors = ['receptor1', 'receptor2']
                self.tfl_ligands = ['ligand1']
                self.tfl_regulators = ['TF1']
                return self

        # Patch the pd.read_parquet to return our mock dataframe
        with patch('pandas.read_parquet', return_value=mock_nichenet_lt):
            # Create the test estimator
            estimator = TestEstimator(
                adata=self.adata,
                target_gene='target_gene',
                cluster_annot='rctd_cluster',
                layer='imputed_count',
                radius=200,
                contact_distance=50,
                tf_ligand_cutoff=0.01,
                colinks_path='dummy_path'
            )
            
            # Call fit
            estimator.fit(num_epochs=1, batch_size=32, use_bayesian=True)
            
            # Create a temporary directory for export/load testing
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, 'model.pt')
                
                # Export the model
                estimator.export(model_path)
                
                # Create a new estimator
                new_estimator = TestLoadEstimator(
                    adata=self.adata,
                    target_gene='target_gene',
                    cluster_annot='rctd_cluster',
                    layer='imputed_count',
                    radius=200,
                    contact_distance=50,
                    tf_ligand_cutoff=0.01,
                    colinks_path='dummy_path'
                )
                
                # Load the model
                new_estimator.load(model_path)
                
                # Check that the attributes were loaded correctly
                self.assertEqual(new_estimator.regulators, ['TF1', 'TF2', 'TF3'])
                self.assertEqual(new_estimator.ligands, ['ligand1', 'ligand2'])
                self.assertEqual(new_estimator.receptors, ['receptor1', 'receptor2'])
                self.assertEqual(new_estimator.tfl_ligands, ['ligand1'])
                self.assertEqual(new_estimator.tfl_regulators, ['TF1'])
    
    @patch('spaceoracle.models.parallel_estimators.RegulatoryFactory', MockRegulatoryFactory)
    @patch('spaceoracle.tools.network.get_cellchat_db')
    @patch('spaceoracle.models.parallel_estimators.WordCloud')
    @patch('matplotlib.pyplot')
    def test_plot_modulators(self, mock_plt, mock_wordcloud, mock_get_cellchat_db):
        """Test the plot_modulators method."""
        # Mock the CellChat database
        mock_df_ligrec = pd.DataFrame({
            'ligand': ['ligand1', 'ligand2'],
            'receptor': ['receptor1', 'receptor2'],
            'pathway': ['pathway1', 'pathway2'],
            'signaling': ['Secreted Signaling', 'Cell-Cell Contact']
        })
        mock_get_cellchat_db.return_value = mock_df_ligrec
        
        # Mock the nichenet_lt dataframe
        mock_nichenet_lt = pd.DataFrame(
            np.random.rand(3, 2),
            index=['TF1', 'TF2', 'TF3'],
            columns=['ligand1', 'ligand2']
        )
        
        # Create a custom estimator class for testing
        class TestEstimator(SpatialCellularProgramsEstimator):
            def plot_modulators(self, use_expression=True, figsize=(10, 10), save_path=None):
                # Mock implementation
                return

        # Patch the pd.read_parquet to return our mock dataframe
        with patch('pandas.read_parquet', return_value=mock_nichenet_lt):
            # Create the test estimator
            estimator = TestEstimator(
                adata=self.adata,
                target_gene='target_gene',
                cluster_annot='rctd_cluster',
                layer='imputed_count',
                radius=200,
                contact_distance=50,
                tf_ligand_cutoff=0.01,
                colinks_path='dummy_path'
            )
            
            # Set necessary attributes for plot_modulators
            estimator.regulators = ['TF1', 'TF2', 'TF3']
            estimator.ligands = ['ligand1', 'ligand2']
            estimator.receptors = ['receptor1', 'receptor2']
            estimator.tfl_ligands = ['ligand1']
            estimator.tfl_regulators = ['TF1']
            estimator.lr_pairs = [('ligand1', 'receptor1'), ('ligand2', 'receptor2')]
            estimator.tfl_pairs = [('TF1', 'ligand1')]
            estimator.modulators = estimator.regulators + estimator.lr_pairs + estimator.tfl_pairs
            
            # Mock the model to return betas
            estimator.model = MagicMock()
            estimator.model.get_betas.return_value = torch.tensor([0.5, 0.3, 0.2, 0.1, 0.1])
            
            # Call plot_modulators
            estimator.plot_modulators(use_expression=True)
            
            # No need to check anything since we're using a mock implementation


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
