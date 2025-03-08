# Interactive UMAP Visualization

A modern and elegant Shiny app for visualizing UMAP plots from single-cell data using scanpy.

## Features

- Interactive file upload for h5ad files
- Adjustable point size and transparency
- Color points by any column in `adata.obs`
- Clean and modern interface
- Responsive visualization

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the app:
```bash
shiny run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8000)

3. Upload your h5ad file using the file upload button

4. Adjust the visualization parameters:
   - Point size: Controls the size of the dots in the plot
   - Point transparency: Controls the opacity of the dots
   - Color by: Select any column from your adata.obs to color the points

## Requirements

- Python 3.8+
- Packages listed in requirements.txt

# SPaceTravLR

[![Python Package using Conda](https://github.com/Koushul/SpaceOracle/actions/workflows/python-package-conda.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceOracle/actions/workflows/python-package-conda.yml)



## Poster

![Poster](./notebooks/beta_example.png)



