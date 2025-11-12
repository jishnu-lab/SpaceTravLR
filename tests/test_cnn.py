import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import numpy as np
import pandas as pd
from unittest import TestCase
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from spaceoracle.models.pixel_attention import CellularNicheNetwork
from spaceoracle.models.spatial_map import xyc2spatial_fast

class CellularNicheNetworkTest(TestCase):
    
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Force CPU usage for all operations
        self.device = torch.device('cpu')
        
        # Generate synthetic data for testing
        self.n_samples = 200
        self.n_features = 5
        self.n_modulators = self.n_features
        self.n_clusters = 3
        self.spatial_dim = 8
        
        # Generate regression data
        self.X, self.y, self.coef = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_features,
            n_targets=1,
            noise=0.1,
            coef=True,
            random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Generate synthetic spatial data
        self.spatial_coords = np.random.rand(self.n_samples, 2)  # x, y coordinates
        self.cluster_labels = np.random.randint(0, self.n_clusters, size=self.n_samples)
        
        # Create spatial maps - we'll use a simplified version for testing
        # Instead of using xyc2spatial_fast which creates a 5D tensor, we'll create a 4D tensor
        # that's compatible with the model's conv layers
        self.spatial_maps = np.random.rand(self.n_samples, 1, self.spatial_dim, self.spatial_dim).astype(np.float32)
        
        # Create spatial features (one-hot encoded cluster labels)
        self.spatial_features = np.zeros((self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            self.spatial_features[i, self.cluster_labels[i]] = 1
        
        # Convert to torch tensors
        self.X_tensor = torch.tensor(self.X_scaled, dtype=torch.float32, device=self.device)
        self.y_tensor = torch.tensor(self.y, dtype=torch.float32, device=self.device)
        self.spatial_maps_tensor = torch.tensor(self.spatial_maps, dtype=torch.float32, device=self.device)
        self.spatial_features_tensor = torch.tensor(self.spatial_features, dtype=torch.float32, device=self.device)
        
        # Create model and move to CPU
        self.model = CellularNicheNetwork(
            n_modulators=self.n_modulators,
            spatial_dim=self.spatial_dim,
            n_clusters=self.n_clusters
        ).to(self.device)
        
        # Ensure anchors are on CPU
        self.model.anchors = self.model.anchors.to(self.device)
    
    def test_model_initialization(self):
        """Test that the model initializes correctly with expected parameters."""
        self.assertEqual(self.model.dim, self.n_modulators + 1)
        self.assertEqual(self.model.spatial_dim, self.spatial_dim)
        self.assertEqual(len(self.model.anchors), self.n_modulators + 1)
        
        # Check that the model has the expected components
        self.assertIsNotNone(self.model.conv_layers)
        self.assertIsNotNone(self.model.spatial_features_mlp)
        self.assertIsNotNone(self.model.mlp)
        self.assertIsNotNone(self.model.output_activation)
        
        # Check that anchors are on the correct device
        self.assertEqual(self.model.anchors.device.type, 'cpu')
    
    def test_get_betas(self):
        """Test the get_betas method returns tensors of the expected shape."""
        # Get betas
        betas = self.model.get_betas(self.spatial_maps_tensor, self.spatial_features_tensor)
        
        # Check shape
        self.assertEqual(betas.shape, (self.n_samples, self.n_modulators + 1))
        
        # Check that betas are finite
        self.assertTrue(torch.all(torch.isfinite(betas)))
    
    def test_predict_y(self):
        """Test the predict_y static method."""
        # Create dummy betas
        dummy_betas = torch.ones((self.n_samples, self.n_modulators + 1), device=self.device)
        
        # Predict y
        y_pred = CellularNicheNetwork.predict_y(self.X_tensor, dummy_betas)
        
        # Check shape
        self.assertEqual(y_pred.shape, (self.n_samples,))
        
        # Check that predictions are finite
        self.assertTrue(torch.all(torch.isfinite(y_pred)))
        
        # For constant betas of 1, y_pred should be sum of features + intercept
        expected_y = self.X_tensor.sum(dim=1) + 1
        self.assertTrue(torch.allclose(y_pred, expected_y))
    
    def test_forward(self):
        """Test the forward method of the model."""
        # Forward pass
        y_pred = self.model(self.spatial_maps_tensor, self.X_tensor, self.spatial_features_tensor)
        
        # Check shape
        self.assertEqual(y_pred.shape, (self.n_samples,))
        
        # Check that predictions are finite
        self.assertTrue(torch.all(torch.isfinite(y_pred)))
    
    def test_training_loop(self):
        """Test that the model can be trained on a toy regression problem."""
        # Split data into train and test sets
        X_train, X_test, y_train, y_test, maps_train, maps_test, features_train, features_test = train_test_split(
            self.X_tensor, self.y_tensor, 
            self.spatial_maps_tensor, 
            self.spatial_features_tensor,
            test_size=0.2, random_state=42
        )
        
        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Train for a few epochs
        n_epochs = 10
        initial_loss = None
        final_loss = None
        
        for epoch in range(n_epochs):
            # Forward pass
            y_pred = self.model(maps_train, X_train, features_train)
            
            # Compute loss
            loss = criterion(y_pred, y_train)
            
            if epoch == 0:
                initial_loss = loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
        
        # Check that loss decreased during training
        self.assertLess(final_loss, initial_loss)
        
        # Evaluate on test set
        with torch.no_grad():
            y_pred_test = self.model(maps_test, X_test, features_test)
            test_loss = criterion(y_pred_test, y_test).item()
        
        # Check that test loss is finite
        self.assertTrue(np.isfinite(test_loss))
    
    def test_sklearn_regression_integration(self):
        """Test integration with scikit-learn regression metrics."""
        # Generate a new regression dataset
        X, y = make_regression(
            n_samples=100,
            n_features=3,
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        
        # Scale features
        X_scaled = StandardScaler().fit_transform(X)
        
        # Generate synthetic spatial data
        spatial_maps = np.random.rand(100, 1, 8, 8).astype(np.float32)
        
        # Create spatial features
        spatial_features = np.zeros((100, 3))
        cluster_labels = np.random.randint(0, 3, size=100)
        for i in range(100):
            spatial_features[i, cluster_labels[i]] = 1
        
        # Convert to torch tensors on CPU
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        spatial_maps_tensor = torch.tensor(spatial_maps, dtype=torch.float32, device=self.device)
        spatial_features_tensor = torch.tensor(spatial_features, dtype=torch.float32, device=self.device)
        
        # Create and train model on CPU
        model = CellularNicheNetwork(n_modulators=3, spatial_dim=8, n_clusters=3).to(self.device)
        model.anchors = model.anchors.to(self.device)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train for a few epochs
        for epoch in range(20):
            y_pred = model(spatial_maps_tensor, X_tensor, spatial_features_tensor)
            loss = criterion(y_pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Make predictions
        with torch.no_grad():
            y_pred = model(spatial_maps_tensor, X_tensor, spatial_features_tensor).cpu().numpy()
        
        # Calculate scikit-learn metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Check that metrics are reasonable
        self.assertTrue(np.isfinite(mse))
        self.assertTrue(np.isfinite(r2))
        
        # After training, R² should be positive (model explains some variance)
        self.assertGreater(r2, 0)
    
    def test_from_pretrained(self):
        """Test the from_pretrained class method."""
        # Create a model to use as pretrained
        pretrained_model = CellularNicheNetwork(
            n_modulators=self.n_modulators,
            spatial_dim=self.spatial_dim,
            n_clusters=self.n_clusters
        ).to(self.device)
        
        # Create a new model from the pretrained one
        new_model = CellularNicheNetwork.from_pretrained(
            trained_model=pretrained_model,
            n_modulators=self.n_modulators,
            spatial_dim=self.spatial_dim,
            n_clusters=self.n_clusters
        ).to(self.device)
        
        # Check that the conv_layers have the same weights
        for (name1, param1), (name2, param2) in zip(
            pretrained_model.conv_layers.named_parameters(),
            new_model.conv_layers.named_parameters()
        ):
            self.assertTrue(torch.allclose(param1, param2))
    
    def test_different_input_shapes(self):
        """Test the model with different input shapes."""
        # Test with different batch sizes
        for batch_size in [1, 10, 50]:
            # Create inputs with the specified batch size
            X = torch.randn(batch_size, self.n_features, device=self.device)
            spatial_maps = torch.randn(batch_size, 1, self.spatial_dim, self.spatial_dim, device=self.device)
            spatial_features = torch.zeros(batch_size, self.n_clusters, device=self.device)
            for i in range(batch_size):
                spatial_features[i, np.random.randint(0, self.n_clusters)] = 1
            
            # Forward pass
            y_pred = self.model(spatial_maps, X, spatial_features)
            
            # Check shape
            self.assertEqual(y_pred.shape, (batch_size,))
            
            # Check that predictions are finite
            self.assertTrue(torch.all(torch.isfinite(y_pred)))
        
        # Test with different spatial dimensions
        # Note: The model architecture requires spatial dimensions to be at least 8
        # due to the 3 max pooling layers with kernel_size=2
        for spatial_dim in [8, 16, 32]:
            # Create a new model with the specified spatial dimension
            model = CellularNicheNetwork(
                n_modulators=self.n_modulators,
                spatial_dim=spatial_dim,
                n_clusters=self.n_clusters
            ).to(self.device)
            model.anchors = model.anchors.to(self.device)
            
            # Create inputs with the specified spatial dimension
            X = torch.randn(10, self.n_features, device=self.device)
            spatial_maps = torch.randn(10, 1, spatial_dim, spatial_dim, device=self.device)
            spatial_features = torch.zeros(10, self.n_clusters, device=self.device)
            for i in range(10):
                spatial_features[i, np.random.randint(0, self.n_clusters)] = 1
            
            # Forward pass
            y_pred = model(spatial_maps, X, spatial_features)
            
            # Check shape
            self.assertEqual(y_pred.shape, (10,))
            
            # Check that predictions are finite
            self.assertTrue(torch.all(torch.isfinite(y_pred)))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        # Create inputs that require gradients
        X = torch.randn(10, self.n_features, device=self.device, requires_grad=True)
        spatial_maps = torch.randn(10, 1, self.spatial_dim, self.spatial_dim, device=self.device, requires_grad=True)
        
        # For spatial_features, we need to create it without requiring gradients initially
        # and then create a one-hot encoding
        cluster_indices = torch.randint(0, self.n_clusters, (10,), device=self.device)
        spatial_features = torch.zeros(10, self.n_clusters, device=self.device)
        spatial_features.scatter_(1, cluster_indices.unsqueeze(1), 1)
        # Now we can set requires_grad
        spatial_features = spatial_features.detach().requires_grad_(True)
        
        # Forward pass
        y_pred = self.model(spatial_maps, X, spatial_features)
        
        # Create a dummy target
        y_target = torch.randn(10, device=self.device)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(y_pred, y_target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed for all parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                # Some gradients might be zero, so we can't check that all are non-zero
                # Just check that at least some gradients are non-zero
                if 'bias' not in name:  # Skip bias parameters as they might have zero gradients
                    self.assertFalse(torch.all(param.grad == 0))
        
        # Check that gradients are computed for inputs
        self.assertIsNotNone(X.grad)
        self.assertIsNotNone(spatial_maps.grad)
        self.assertIsNotNone(spatial_features.grad)
    
    def test_different_hyperparameters(self):
        """Test the model with different hyperparameters."""
        # Test with different numbers of modulators
        for n_modulators in [1, 10, 20]:
            # Create a new model with the specified number of modulators
            model = CellularNicheNetwork(
                n_modulators=n_modulators,
                spatial_dim=self.spatial_dim,
                n_clusters=self.n_clusters
            ).to(self.device)
            model.anchors = model.anchors.to(self.device)
            
            # Create inputs with the specified number of modulators
            X = torch.randn(10, n_modulators, device=self.device)
            spatial_maps = torch.randn(10, 1, self.spatial_dim, self.spatial_dim, device=self.device)
            spatial_features = torch.zeros(10, self.n_clusters, device=self.device)
            for i in range(10):
                spatial_features[i, np.random.randint(0, self.n_clusters)] = 1
            
            # Forward pass
            y_pred = model(spatial_maps, X, spatial_features)
            
            # Check shape
            self.assertEqual(y_pred.shape, (10,))
            
            # Check that predictions are finite
            self.assertTrue(torch.all(torch.isfinite(y_pred)))
        
        # Test with different numbers of clusters
        for n_clusters in [1, 5, 10]:
            # Create a new model with the specified number of clusters
            model = CellularNicheNetwork(
                n_modulators=self.n_modulators,
                spatial_dim=self.spatial_dim,
                n_clusters=n_clusters
            ).to(self.device)
            model.anchors = model.anchors.to(self.device)
            
            # Create inputs with the specified number of clusters
            X = torch.randn(10, self.n_modulators, device=self.device)
            spatial_maps = torch.randn(10, 1, self.spatial_dim, self.spatial_dim, device=self.device)
            spatial_features = torch.zeros(10, n_clusters, device=self.device)
            for i in range(10):
                spatial_features[i, np.random.randint(0, n_clusters)] = 1
            
            # Forward pass
            y_pred = model(spatial_maps, X, spatial_features)
            
            # Check shape
            self.assertEqual(y_pred.shape, (10,))
            
            # Check that predictions are finite
            self.assertTrue(torch.all(torch.isfinite(y_pred)))
    
    def test_custom_anchors(self):
        """Test the model with custom anchors."""
        # Create custom anchors
        custom_anchors = np.random.rand(self.n_modulators + 1).astype(np.float32)
        
        # Create a new model with custom anchors
        model = CellularNicheNetwork(
            n_modulators=self.n_modulators,
            anchors=custom_anchors,
            spatial_dim=self.spatial_dim,
            n_clusters=self.n_clusters
        ).to(self.device)
        model.anchors = model.anchors.to(self.device)
        
        # Check that the anchors are set correctly
        self.assertTrue(torch.allclose(model.anchors, torch.tensor(custom_anchors, device=self.device)))
        
        # Forward pass
        y_pred = model(self.spatial_maps_tensor, self.X_tensor, self.spatial_features_tensor)
        
        # Check shape
        self.assertEqual(y_pred.shape, (self.n_samples,))
        
        # Check that predictions are finite
        self.assertTrue(torch.all(torch.isfinite(y_pred)))
    
    def test_model_reproducibility(self):
        """Test that the model produces the same outputs given the same inputs and seed."""
        # Set seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create two identical models
        model1 = CellularNicheNetwork(
            n_modulators=self.n_modulators,
            spatial_dim=self.spatial_dim,
            n_clusters=self.n_clusters
        ).to(self.device)
        model1.anchors = model1.anchors.to(self.device)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        model2 = CellularNicheNetwork(
            n_modulators=self.n_modulators,
            spatial_dim=self.spatial_dim,
            n_clusters=self.n_clusters
        ).to(self.device)
        model2.anchors = model2.anchors.to(self.device)
        
        # Check that the models have the same parameters
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            self.assertTrue(torch.allclose(param1, param2))
        
        # Create inputs
        X = torch.randn(10, self.n_modulators, device=self.device)
        spatial_maps = torch.randn(10, 1, self.spatial_dim, self.spatial_dim, device=self.device)
        spatial_features = torch.zeros(10, self.n_clusters, device=self.device)
        for i in range(10):
            spatial_features[i, np.random.randint(0, self.n_clusters)] = 1
        
        # Forward pass
        with torch.no_grad():
            y_pred1 = model1(spatial_maps, X, spatial_features)
            y_pred2 = model2(spatial_maps, X, spatial_features)
        
        # Check that the predictions are the same
        self.assertTrue(torch.allclose(y_pred1, y_pred2))
