from .parallel_estimators import *
from .pixel_attention import CellularNicheNetwork

class PrefeaturizedTensorDataset(Dataset):
    def __init__(self, sp_maps, X_cell, y_cell):
        self.sp_maps = sp_maps
        self.X_cell = X_cell
        self.y_cell = y_cell

    def __len__(self):
        return len(self.X_cell)

    def __getitem__(self, idx):
        sp_map = self.sp_maps[idx, :, :]

        return (
            torch.from_numpy(sp_map.copy()).float(),
            torch.from_numpy(self.X_cell[idx]).float(),
            torch.from_numpy(np.array(self.y_cell[idx])).float()
        )
      
class PrefeaturizedNicheNetwork(CellularNicheNetwork):
   
    def get_betas(self, spatial_maps):
        out = self.conv_layers(spatial_maps)
        # sp_out = self.spatial_features_mlp(spatial_features)
        # out = out+sp_out
        betas = self.mlp(out)
        betas = self.output_activation(betas) * 1.5

        return betas*self.anchors
    
    def forward(self, spatial_maps, inputs_x):
        betas = self.get_betas(spatial_maps)
        y_pred = self.predict_y(inputs_x, betas)
        
        return y_pred

class PrefeaturizedCellularProgramsEstimator(SpatialCellularProgramsEstimator):

    def __init__(self, adata, target_gene, 
            cluster_annot='cell_type_int', layer='imputed_count', 
            radius=100, contact_distance=30, use_ligands=True,
            tf_ligand_cutoff=0.01, receptor_thresh=0.1,
            regulators=None, grn=None, colinks_path=None, scale_factor=1,
            sp_maps_key='COVET_SQRT'):

        assert sp_maps_key in adata.obsm.keys(), f'adata.obsm does not contain {sp_maps_key}'
        sp_maps = adata.obsm[sp_maps_key]

        super().__init__(adata, target_gene, spatial_dim=sp_maps.shape[1], 
            cluster_annot=cluster_annot, layer=layer, 
            radius=radius, contact_distance=contact_distance, use_ligands=use_ligands,
            tf_ligand_cutoff=tf_ligand_cutoff, receptor_thresh=receptor_thresh,
            regulators=regulators, grn=grn, colinks_path=colinks_path, scale_factor=scale_factor)
        
        self.sp_maps_key = sp_maps_key


    def init_data(self):
        '''
        Initialize the data for the estimator, without processing spatial maps
        '''

        lr_info = self.check_LR_properties(self.adata, self.layer)
        counts_df, cell_thresholds = lr_info

        if not all(
            hasattr(self.adata.uns, attr) 
            for attr in ['received_ligands', 'received_ligands_tfl']
        ):
            self.adata = init_received_ligands(
                self.adata,
                radius=self.radius, 
                contact_distance=self.contact_distance, 
                cell_threshes=cell_thresholds
            )

        if len(self.lr['pairs']) > 0:
            
            self.adata.uns['ligand_receptor'] = self.ligands_receptors_interactions(
                self.adata.uns['received_ligands'][self.ligands], 
                get_filtered_df(counts_df, cell_thresholds, self.receptors)[self.receptors]
            )

        else:
            self.adata.uns['received_ligands'] = pd.DataFrame(index=self.adata.obs.index)
            self.adata.uns['ligand_receptor'] = pd.DataFrame(index=self.adata.obs.index)


        if len(self.tfl_pairs) > 0:

            self.adata.uns['ligand_regulator'] = self.ligand_regulators_interactions(
                self.adata.uns['received_ligands_tfl'][self.tfl_ligands], 
                self.adata.to_df(layer=self.layer)[self.tfl_regulators]
            )
        else:
            self.adata.uns['ligand_regulator'] = pd.DataFrame(index=self.adata.obs.index)

        cluster_labels = np.array(self.adata.obs[self.cluster_annot])

        self.train_df = self.adata.to_df(layer=self.layer)[
            [self.target_gene]+self.regulators] \
            .join(self.adata.uns['ligand_receptor']) \
            .join(self.adata.uns['ligand_regulator'])
        
        # Filter modulators
        low_std = self.train_df.join(
            self.adata.obs['cell_type_int']
        ).groupby('cell_type_int').std().max(0) < 1e-8
        low_std = low_std.loc[self.train_df.columns]
        
        self.train_df = self.train_df.loc[:, ~low_std]
        self.lr_pairs = self.lr_pairs[self.lr_pairs.isin(self.train_df.columns)]
        self.tfl_pairs = [i for i in self.tfl_pairs if i in self.train_df.columns]
        
        self.ligands = []
        self.receptors = []
        self.tfl_regulators = []
        self.tfl_ligands = []
        
        for i in self.lr_pairs:
            lig, rec = i.split('$')
            self.ligands.append(lig)
            self.receptors.append(rec)
            
        for i in self.tfl_pairs:
            lig, reg = i.split('#')
            self.tfl_ligands.append(lig)
            self.tfl_regulators.append(reg)
            
            
        self.modulators = self.regulators + list(self.lr_pairs) + self.tfl_pairs
        self.modulators_genes = list(np.unique(
            self.regulators+self.ligands+self.receptors+self.tfl_regulators+self.tfl_ligands))

        assert len(self.ligands) == len(self.receptors)

        X = self.train_df.drop(columns=[self.target_gene]).values
        y = self.train_df[self.target_gene].values
        
        sp_maps = self.adata.obsm[self.sp_maps_key]

        assert sp_maps.shape[0] == X.shape[0] == y.shape[0] == len(cluster_labels)
        return sp_maps, X, y, cluster_labels

    def fit(self, num_epochs=100, threshold_lambda=1e-6, learning_rate=5e-3, batch_size=512, 
            pbar=None, estimator='lasso',
            score_threshold=0.2):
        
        sp_maps, X, y, cluster_labels = self.init_data()


        assert estimator in ['lasso', 'bayesian', 'ard']
        self.estimator = estimator
        self.models = {}
        self.Xn = X
        self.yn = y
        self.sp_maps = sp_maps
        self.cell_indices = self.adata.obs.index.copy()
        self.cluster_labels = cluster_labels

        if pbar is None:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=sp_maps.shape[0]*num_epochs, 
                desc='Estimating Spatial Betas', unit='cells',
                color='green',
                auto_refresh=True
            )

        if num_epochs:
            print(f'Fitting {self.target_gene} with {len(self.modulators)} modulators')
            print(f'\t{len(self.regulators)} Transcription Factors')
            print(f'\t{len(self.lr_pairs)} Ligand-Receptor Pairs')
            print(f'\t{len(self.tfl_pairs)} TranscriptionFactor-Ligand Pairs')

        for cluster in np.unique(cluster_labels):
            mask = cluster_labels == cluster
            X_cell, y_cell = self.Xn[mask], self.yn[mask]

            if self.estimator == 'ard': 
                """
                ARD allocates a n_samples * n_samples matrix so isn't very scalable
                """
                m = ARDRegression(threshold_lambda=threshold_lambda)
                m.fit(X_cell, y_cell)
                y_pred = m.predict(X_cell)
                r2 = r2_score(y_cell, y_pred)
                _betas = np.hstack([m.intercept_, m.coef_])
                coefs = None

            elif self.estimator == 'bayesian':
                m = BayesianRidge()
                m.fit(X_cell, y_cell)
                y_pred = m.predict(X_cell)
                r2 = r2_score(y_cell, y_pred)
                _betas = np.hstack([m.intercept_, m.coef_])

            elif self.estimator == 'lasso':
                groups = [1]*len(self.regulators) + [2]*len(self.lr_pairs) + [3]*len(self.tfl_pairs)
                groups = np.array(groups)
                gl = GroupLasso(
                    groups=groups,
                    group_reg=threshold_lambda,
                    l1_reg=1e-9,
                    frobenius_lipschitz=True,
                    scale_reg="inverse_group_size",
                    # subsampling_scheme=1,
                    # supress_warning=True,
                    n_iter=1000,
                    tol=1e-5,
                )
                gl.fit(X_cell, y_cell)
                y_pred = gl.predict(X_cell)
                coefs = gl.coef_.flatten()
                _betas = np.hstack([gl.intercept_, coefs])
                r2 = r2_score(y_cell, y_pred)
            

            loader = DataLoader(
                PrefeaturizedTensorDataset(
                    sp_maps[mask],
                    X_cell,
                    y_cell
                ),
                batch_size=batch_size, shuffle=True
            )

            assert _betas.shape[0] == len(self.modulators)+1

            model = PrefeaturizedNicheNetwork(
                    n_modulators = len(self.modulators), 
                    anchors=_betas,
                    spatial_dim=self.spatial_dim,
                    n_clusters=self.n_clusters
                ).to(self.device)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                all_y_true = []
                all_y_pred = []
                
                for batch in loader:
                    spatial_maps, inputs, targets = [b.to(device) for b in batch]
                    
                    import pdb; pdb.set_trace()

                    optimizer.zero_grad()
                    outputs = model(spatial_maps, inputs)
                    loss = criterion(outputs, targets)
                    loss += torch.mean(outputs.mean(0) - model.anchors) * 1e-3
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    all_y_true.extend(targets.cpu().detach().numpy())
                    all_y_pred.extend(outputs.cpu().detach().numpy())

                    pbar.desc = f'{self.target_gene} | {cluster+1}/{self.n_clusters}'
                    pbar.update(len(targets))

            if num_epochs:
                score = r2_score(all_y_true, all_y_pred)
                if score < score_threshold: 
                    # no point in predicting betas if we do it poorly
                    model.anchors = model.anchors*0.0
                    print(f'{cluster}: x.xxxx | {r2:.4f}')
                    
                  
                else:
                    print(f'{cluster}: {score:.4f} | {r2:.4f}')
            
            self.models[cluster] = model