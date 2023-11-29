# Contact tracing for HIV-1 phylodynamics and inference

Code from _Kupperman et al._ "TITLE" (2023). Under review.

This repository contains codes and supplementary materials for the manuscript using R and Python 3. Codes to generate figures 4 and 7 in the manuscript.


## How to use

Install requirements into a virtual environment:

```bash
conda create -n ContactTracing python=3.10 numpy scipy matplotlib pandas seaborn r-base r-future r-ape r-treebalance r-phangorn -c conda-forge
```
Install SEEPS from Github using devtools. This may require installing additional system dependencies.
```bash
> Rscript -e "install.packages('devtools')"
> Rscript -e "devtools::install_git('git@github.com:MolEvolEpid/SEEPS.git', ref='feature/ref_branching_models')"
```

### Figure 4

First we use SEEPS to simulate data.

```bash
Rscript src/R/generate_data_5a_test.R
Rscript src/R/generate_data_5d_test.R
```

Simulation outputs are stored in `synth_data/SEEPS_5#/##`.
Large simulated data files are ommited from the repository, but can be generated using the above commands.

Plotting code for double side-by-side violins is stored in the notebook `notebooks/TimeSeries_comparisons.ipynb`.

### Figure 7

GenBank accession numbers for the sequences used in the analysis are stored in `raw_data/ids.txt`. We have included the subtree sampled statistics in `raw_data/`, as well as the sample year distribution.

Simulation functions that use SEEPS are stored in `src/R/Sampler_EU.R` and `src/R/Sampler_SE.R`. Driver code can be found in the notebooks `notebooks/Learning_CT_p_EU.ipynb` and `notebooks/Learning_CT_p_SE.ipynb`.
