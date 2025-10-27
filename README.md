[![Tests](https://github.com/Koushul/SpaceOracle/actions/workflows/python-package-conda.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceOracle/actions/workflows/python-package-conda.yml)

# Why SpaceTravLR 🌔️ ?

**SpaceTravLR** (**S**patially perturbing **T**ranscription factors, **L**igands & **R**eceptors)

<p align="center">
  <img src="./assets/overview.svg" alt="overview" style="width:1200px"/>
</p>

Cell intrinsic regulation is captured through transcription factor (TF) terms, while signaling is modeled via distance-weighted ligand expression from neighboring sender cells to each receiver cell Specifically, signaling is captured based on both ligand-receptor and ligand-TF associations. To integrate spatial information while maintaining biological interpretability, SpaceTravLR leverages convolutional neural networks to generate a sparse graph with differentiable edges. This architecture enables signals to propagate both within cells through regulatory edges and between cells through ligand–mediated connections, and is mathematically computed by efficient, gradient-based perturbation analysis via the chain rule. 








## Core Features
- inferring functional cell-cell communications events
- *in-silico* modeling of functional and spatial reprogramming following perturbations
- identifying spatial domains and functional microniches and their driver genes

Read more on our [documentation website](https://).



##  Quick start

Install dependencies
```bash
pip install -r requirements.txt
```

Load the example [Slide-tags]((https://www.nature.com/articles/s41586-023-06837-4)) Human Tonsil data.

```python
adata = sc.read_h5ad('data/snrna_germinal_center.h5ad')
```

Create a SpaceTravLR model
```python
import SpaceTravLR

space_lab = SpaceTravLR()
space_lab.fit(
    adata,
    radius=250, 
    num_epochs=100, 
    learning_rate=5e-3, 
)
```

```python
space_lab.plot()
```

##  Example

##  Outputs

##  Results



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