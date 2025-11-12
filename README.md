[![Tests](https://github.com/Koushul/SpaceOracle/actions/workflows/python-package-conda.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceOracle/actions/workflows/python-package-conda.yml)

# Why SpaceTravLR 🌔️ ?

**SpaceTravLR** (**S**patially perturbing **T**ranscription factors, **L**igands & **R**eceptors)



<p align="center">
  <img src="./assets/overview.svg" alt="overview" style="width:1200px"/>
</p>

Cell intrinsic regulation is captured through transcription factor (TF) terms, while signaling is modeled via distance-weighted ligand expression from neighboring sender cells to each receiver cell Specifically, signaling is captured based on both ligand-receptor and ligand-TF associations. To integrate spatial information while maintaining biological interpretability, SpaceTravLR leverages convolutional neural networks to generate a sparse graph with differentiable edges. This architecture enables signals to propagate both within cells through regulatory edges and between cells through ligand–mediated connections, and is mathematically computed by efficient, gradient-based perturbation analysis via the chain rule. 

<p align="center">
  <img src="./assets/model.svg" alt="overview" style="width:1200px"/>
</p>










## Core Features
- inferring functional cell-cell communications events
- *in-silico* modeling of functional and spatial reprogramming following perturbations
- identifying spatial domains and functional microniches and their driver genes

Read more on our [documentation website](https://).



##  Quick start

Make & Sync your Environment the modern way

~~pip install -r requirements.txt~~

```bash
uv env
source .venv/bin/activate
uv sync
```

Load the example [Slide-tags]((https://www.nature.com/articles/s41586-023-06837-4)) Human Tonsil data.

```python
adata = sc.read_h5ad('data/snrna_germinal_center.h5ad')
```

Create a SpaceShip
```python
from SpaceTravLR.spaceship import SpaceShip

spacetravlr = SpaceShip(name='myTonsil').setup_(adata)

assert spacetravlr.is_everything_ok()

spacetravlr.spawn_worker(partition='l40s')
```

##  Outputs
<pre>
output/
├── input_data/
│   ├── _adata.h5ad
│   ├── celloracle_links.pkl
│   ├── communication.pkl
│   ├── LRs.parquet
├── betadata/
│   ├── PAX5_betadata.parquet
│   ├── FOXO1_betadata.parquet
│   ├── CD79A_betadata.parquet
│   ├── ...
│   ├── IL21_betadata.parquet
│   ├── IL4_betadata.parquet
│   ├── CCR4_betadata.parquet
├── logs/
│   ├── training_TIMESTAMP.log

</pre>

##  Results

<p align="center">
  <img src="./assets/GC_FOXO1_KO.svg" alt="overview" style="width:1200px"/>
</p>

##  Parallel Training & Inference
SpaceTravLR implements a scalable pipeline leveraging high performance computing for parallelized tensor computations to estimate the target gene expression based on spatially varying regulatory and signaling dynamics. 


##  FAQ
<details>
<summary><strong>How long does SpaceTravLR take to train?</strong></summary>
</details>


<details>
<summary><strong>Do I need paired ATAC-seq data?</strong></summary>
</details>



## Citation

If you find SpaceTravLR useful in your research or projects, please cite our paper:
```