# MPC-Rank experimental code

This repository contains the programs used to generate the numerical results for the MPC-Rank study. It does not include raw third-party datasets, recorded result files, or the separate publication-figure generator.

## Contents

- `code/metapath_coupling_experiment.py`: model, source-data loaders, real-data experiments, support sweeps, and long-tail-layer experiments.
- `code/synthetic_boundary_experiment.py`: synthetic comparison-data generation and mechanism experiments. This script imports the model from the first script, so keep both files in the `code/` directory.
- `requirements.txt`: Python dependencies.

The real-data experiment script also writes some diagnostic plots when run. Its plotting functions are embedded in the experiment program; the separate program used to make publication figures is not included here.

## Requirements

Use Python 3.10 or later. Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Source data

Obtain the original datasets from their providers and follow their terms:

- Breakfast: https://github.com/PrefLib/PrefLib-Data (file `datasets/00035 - breakfast/00035-00000001.csv`)
- SUSHI3: https://www.kamishima.net/sushi/ (the `sushi3-2016` archive)
- MovieLens 100K: https://grouplens.org/datasets/movielens/100k/ (the `ml-100k` archive)

Place the required files under `data/` as follows, or set `MPC_DATA_ROOT` to another directory with these same three subdirectories:

```text
data/
  breakfast/00035-00000001.csv
  sushi3-2016/sushi3.udata
  sushi3-2016/sushi3a.5000.10.order
  ml-100k/u.data
  ml-100k/u.user
```

Raw SUSHI3 and MovieLens files are not redistributed in this repository.

## Run experiments

From the repository root:

```bash
python code/metapath_coupling_experiment.py --seeds 12 --steps 750 --out results/real
python code/synthetic_boundary_experiment.py --seeds 30 --steps 300
```

The first command requires the three source datasets and writes CSV results and diagnostic plots to `results/real/`. The second generates its comparison data internally and writes CSV results to `results/`. Full runs may take considerable time. For a small execution check, use `--smoke`; those outputs are not the manuscript results.

This repository provides the experimental programs. It does not claim that the full experiments were independently rerun during repository preparation. The original experiment manifests did not record every package version.
