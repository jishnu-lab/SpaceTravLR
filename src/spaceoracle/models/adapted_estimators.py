from .parallel_estimators import *
from .pixel_attention import CellularNicheNetwork
from ..tools.network import expand_paired_interactions
from ..tools.utils import is_mouse_data

import torch
import torch.nn as nn 
import commot as ct
import itertools


class PrefeaturizedTensorDataset(Dataset):
    def __init__(self, sp_maps, X_cell, y_cell):
        self.sp_maps = sp_maps
        self.X_cell = X_cell
        self.y_cell = y_cell

    def __len__(self):
        return len(self.X_cell)

    def __getitem__(self, idx):
        sp_map = self.sp_maps[idx, :]
        sp_map = np.expand_dims(sp_map, axis=0)

        return (
            torch.from_numpy(sp_map.copy()).float(),
            torch.from_numpy(self.X_cell[idx]).float(),
            torch.from_numpy(np.array(self.y_cell[idx])).float()
        )
      
class PrefeaturizedNicheNetwork(nn.Module):

    def __init__(self, n_modulators, spatial_dim=64**2):
        super().__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.spatial_dim = spatial_dim
        self.dim = n_modulators+1

        self.mlp = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.PReLU(init=0.1),
            nn.Linear(64, self.dim)
        )
        
        self.output_activation = nn.Sigmoid()
        
    @staticmethod
    def predict_y(inputs_x, betas):
        return torch.matmul(
                inputs_x.unsqueeze(1), 
                betas[:, 1:].unsqueeze(2)
            ).squeeze(1).squeeze(1) + \
                betas[:, 0]

    def get_betas(self, spatial_maps):
        
        n_samples = spatial_maps.shape[0]
        betas = self.mlp(spatial_maps.reshape(n_samples, self.spatial_dim))
        betas = self.output_activation(betas) * 1.5

        return betas
    
    def forward(self, spatial_maps, inputs_x):

        betas = self.get_betas(spatial_maps)
        y_pred = self.predict_y(inputs_x, betas)
        
        return y_pred

class PrefeaturizedCellularProgramsEstimator(SpatialCellularProgramsEstimator):

    def __init__(self, adata, target_gene, layer='imputed_count', 
            radius=100, contact_distance=30, use_ligands=True,
            tf_ligand_cutoff=0.01, receptor_thresh=0.1,
            regulators=None, grn=None, colinks_path=None, scale_factor=1,
            sp_maps_key='COVET_SQRT'):

        assert sp_maps_key in adata.obsm.keys(), f'adata.obsm does not contain {sp_maps_key}'
        sp_maps = adata.obsm[sp_maps_key].reshape(adata.n_obs, -1)

        super().__init__(adata, target_gene, spatial_dim=sp_maps.shape[1], layer=layer, 
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

        self.train_df = self.adata.to_df(layer=self.layer)[
            [self.target_gene]+self.regulators] \
            .join(self.adata.uns['ligand_receptor']) \
            .join(self.adata.uns['ligand_regulator'])
        
        if len(self.lr_pairs) > 0:
            self.lr_pairs = self.lr_pairs[self.lr_pairs.isin(self.train_df.columns)]
        if len(self.tfl_pairs) > 0:
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

        assert sp_maps.shape[0] == X.shape[0] == y.shape[0] 
        return sp_maps, X, y 

    def fit(self, num_epochs=100, learning_rate=5e-3, batch_size=512, pbar=None):
        
        sp_maps, X, y = self.init_data()
        print(f'Using {self.sp_maps_key} as spatial maps')

        self.models = {}
        self.Xn = X
        self.yn = y
        self.sp_maps = sp_maps
        self.cell_indices = self.adata.obs.index.copy()

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

        X_cell, y_cell = self.Xn, self.yn

        loader = DataLoader(
            PrefeaturizedTensorDataset(
                sp_maps,
                X_cell,
                y_cell
            ),
            batch_size=batch_size, shuffle=True
        )

        model = PrefeaturizedNicheNetwork(
                n_modulators = len(self.modulators), 
                spatial_dim=self.spatial_dim
            ).to(self.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        
        # Early stopping parameters
        patience = 10
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            all_y_true = []
            all_y_pred = []
            
            for batch in loader:
                spatial_maps, inputs, targets = [b.to(device) for b in batch]
                
                optimizer.zero_grad()
                outputs = model(spatial_maps, inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()
                
                epoch_loss += loss.item()
                all_y_true.extend(targets.cpu().detach().numpy())
                all_y_pred.extend(outputs.cpu().detach().numpy())

                pbar.desc = f'{self.target_gene}'
                pbar.update(len(targets))
    
            # Early stopping check
            avg_epoch_loss = epoch_loss / len(loader)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch > 50 and patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

        if num_epochs:
            score = r2_score(all_y_true, all_y_pred)
            print(f'{score:.4f}')
        
        self.models = model
    
    @torch.no_grad()
    def get_betas(self):
        index_tracker = []
        betas = []
        indices = self.cell_indices
        index_tracker.extend(indices)
        sp_maps = torch.from_numpy(
            self.sp_maps).float().unsqueeze(1)
        b = self.models.get_betas(
            sp_maps.to(self.device),
        ).cpu().numpy()
        betas.extend(b)

        return pd.DataFrame(
            betas, 
            index=index_tracker, 
            columns=['beta0']+['beta_'+i for i in self.modulators]
        ).reindex(self.adata.obs.index)



class GeneProgramsEstimator(PrefeaturizedCellularProgramsEstimator):

    def __init__(self, adata, target_gene, mgs, layer='imputed_count', 
                 tf_ligand_cutoff=0.01, radius=100, contact_distance=30, scale_factor=1,
                 sp_maps_key='scGPT', use_ligands=False):

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.device = device
        self.adata = adata
        self.radius = radius
        self.contact_distance = contact_distance
        self.layer = layer
        self.mgs = mgs
        self.target_gene = target_gene
        self.tf_ligand_cutoff = tf_ligand_cutoff
        self.sp_maps_key = sp_maps_key
        self.spatial_dim = adata.obsm[sp_maps_key].shape[1]
        self.use_ligands = use_ligands

        ligand_mixtures = self.init_ligands_and_receptors(use_ligands=use_ligands)

        self.tfl_pairs = ligand_mixtures['tfl_pairs']        
        self.tfl_regulators = ligand_mixtures['tfl_regulators']
        self.tfl_ligands = ligand_mixtures['tfl_ligands']
        self.regulators = ligand_mixtures['regulators']

        # Treat all interactions as tfl pairs
        self.ligands = []
        self.receptors = []
        self.lr = {'pairs': []}
        self.lr_pairs = self.lr['pairs']
    

    def init_ligands_and_receptors(self, use_ligands=True):
    
        adata = self.adata
        species = 'mouse' if is_mouse_data(adata) else 'human'
        df_ligrec = ct.pp.ligand_receptor_database(
            database='CellChat', 
            species=species, 
            signaling_type=None
        ) 
        df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']  

        lr = expand_paired_interactions(df_ligrec)
        lr = lr[lr.ligand.isin(adata.var_names) &\
            (lr.receptor.isin(adata.var_names))]
        ligands = np.unique(lr.ligand)

        tfl_pairs = []
        tfl_regulators = []
        tfl_ligands = []

        grns = [genes for module, genes in self.mgs.items() if self.target_gene in genes]

        regulators = list(set(itertools.chain.from_iterable(grns)) - {self.target_gene})

        ligand_mixtures = edict()

        if use_ligands:

            for grn in grns:
                grn = [g for g in grn if g != self.target_gene]
                grn_pairs = pd.DataFrame(list(itertools.combinations(grn, 2)), columns=['gene1', 'gene2'])

                for g1, g2 in grn_pairs.values:
                    if g1 in ligands:
                        tfl_pairs.append(f'{g1}#{g2}')
                        tfl_regulators.append(g2)
                        tfl_ligands.append(g1)
                    if g2 in ligands:
                        tfl_pairs.append(f'{g2}#{g1}')
                        tfl_regulators.append(g1)
                        tfl_ligands.append(g2)

            ligand_mixtures['tfl_pairs'] = tfl_pairs
            ligand_mixtures['tfl_regulators'] = tfl_regulators
            ligand_mixtures['tfl_ligands'] = tfl_ligands

        else:

            ligand_mixtures['tfl_pairs'] = []
            ligand_mixtures['tfl_regulators'] = []
            ligand_mixtures['tfl_ligands'] = []

        ligand_mixtures['regulators'] = regulators
        return ligand_mixtures