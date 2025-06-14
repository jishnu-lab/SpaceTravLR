import enlighten
import os
import pandas as pd
import time
from .models.adapted_estimators import PrefeaturizedCellularProgramsEstimator, GeneProgramsEstimator

from .oracles import SpaceTravLR

class Astronaut(SpaceTravLR):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # This is okay because we init ligands in fit()
        
        for k in ['cell_thresholds', 'received_ligands', 'received_ligands_tfl']:
            if k in self.adata.uns:
                del self.adata.uns[k]

    def run(self, sp_maps_key='COVET_SQRT'):

        _manager = enlighten.get_manager()

        gene_bar = _manager.counter(
            total=len(self.queue.all_genes), 
            desc=f'... initializing ...', 
            unit='genes',
            color='green',
            autorefresh=True,
        )

        train_bar = _manager.counter(
            total=self.adata.shape[0]*self.max_epochs, 
            desc=f'Ready...', unit='cells',
            color='red',
            auto_refresh=True
        )


        while not self.queue.is_empty and not os.path.exists(self.save_dir+'/process.kill'):
            
            # Remove old locks from other models
            self.queue.kill_old_locks()

            gene = next(self.queue)

            estimator = PrefeaturizedCellularProgramsEstimator(
                adata=self.adata,
                target_gene=gene,
                layer=self.layer,
                radius=self.radius,
                contact_distance=self.contact_distance,
                tf_ligand_cutoff=self.tf_ligand_cutoff,
                grn=self.grn,
                sp_maps_key=sp_maps_key
            )
            
            estimator.test_mode = False
            
            if len(estimator.regulators) == 0:
                self.queue.add_orphan(gene)
                continue

            else:
                gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
                gene_bar.desc = f'🕵️️  {self.queue.agents+1} agents'
                gene_bar.refresh()

                if os.path.exists(f'{self.queue.model_dir}/{gene}.lock'):
                    continue

                self.queue.create_lock(gene)

                estimator.fit(
                    num_epochs=self.max_epochs, 
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    pbar=train_bar
                )

                estimator.betadata.to_parquet(f'{self.save_dir}/{gene}_betadata.parquet')

                self.trained_genes.append(gene)
                self.queue.delete_lock(gene)

            gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
            gene_bar.refresh()

            train_bar.count = 0
            train_bar.start = time.time()
            
        gene_bar.desc = 'All done! 🎉️'
        gene_bar.refresh()


class GeneGeneAstronaut(Astronaut):
    def run(self, sp_maps_key='scGPT'):

        _manager = enlighten.get_manager()

        gene_bar = _manager.counter(
            total=len(self.queue.all_genes), 
            desc=f'... initializing ...', 
            unit='genes',
            color='green',
            autorefresh=True,
        )

        train_bar = _manager.counter(
            total=self.adata.shape[0]*self.max_epochs, 
            desc=f'Ready...', unit='cells',
            color='red',
            auto_refresh=True
        )


        while not self.queue.is_empty and not os.path.exists(self.save_dir+'/process.kill'):
            
            # Remove old locks from other models
            self.queue.kill_old_locks()

            gene = next(self.queue)

            estimator = GeneProgramsEstimator(
                adata=self.adata,
                target_gene=gene,
                layer=self.layer,
                radius=self.radius,
                contact_distance=self.contact_distance,
                tf_ligand_cutoff=self.tf_ligand_cutoff,
                mgs=self.grn,
                sp_maps_key=sp_maps_key
            )
            
            estimator.test_mode = False
            
            if len(estimator.regulators) == 0:
                self.queue.add_orphan(gene)
                continue

            else:
                gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
                gene_bar.desc = f'🕵️️  {self.queue.agents+1} agents'
                gene_bar.refresh()

                if os.path.exists(f'{self.queue.model_dir}/{gene}.lock'):
                    continue

                self.queue.create_lock(gene)

                estimator.fit(
                    num_epochs=self.max_epochs, 
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    pbar=train_bar
                )

                estimator.betadata.to_parquet(f'{self.save_dir}/{gene}_betadata.parquet')

                self.trained_genes.append(gene)
                self.queue.delete_lock(gene)

            gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
            gene_bar.refresh()

            train_bar.count = 0
            train_bar.start = time.time()
            
        gene_bar.desc = 'All done! 🎉️'
        gene_bar.refresh()

