
This repository contains the code used for the dataset construction experiments described in this project.

The code was originally developed and tested on Compute Canada / Alliance systems. For public users, we provide a separate installation path using a public `requirements.txt` file.

---
# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

This repository provides two installation options:

1. **Public/local installation** using `requirements.txt`
2. **Compute Canada installation** using `requirements_compute_canada.txt`

Users with no Compute Canada access should use the first option.

Create the environment and install the required packages (by default uses `requirements.txt` ):
```bash

bash install_env.sh
```
This script creates a Python virtual environment named test and installs the packages listed in requirements.txt.
After installation, activate the environment with:
```bash
source test/bin/activate
```

On Windows Git Bash, use:
```bash
source test/Scripts/activate
```

---


# ExperimentalDatasetConstruction

The `ExperimentalDatasetConstruction` folder contains the code used to reproduce the experimental dataset construction experiments using the **sysTEm** dataset as the candidate pool.

This folder contains three subfolders:

- `systemDataCuration`
- `ActiveLearningDatasetConstruction`
- `ResultsAnalysis`

---

## 1. `systemDataCuration`

The `systemDataCuration` folder contains the code and raw data used to curate and featurize the sysTEm dataset.

### `DataCuration_sysTEmDataset.ipynb`

This notebook featurizes the chemical compositions in `sysTEm_dataset.xlsx` and curates the dataset.

Before running the notebook, make sure that:

1. The sysTEm dataset has been downloaded.
2. The local file path to `sysTEm_dataset.xlsx` is correctly specified in the notebook.

The notebook outputs a curated CSV file named:

```text
CLEANED_DATA.csv
```

This file is used as the candidate pool for the active-learning dataset construction experiments.

### `sysTEm_dataset.xlsx`

This is the original raw sysTEm dataset.

The dataset is available from the following GitHub repository:

```text
https://github.com/tankylz/sysTEm_dataset
```

Dataset DOI:

```text
10.26434/chemrxiv-2025-4gxmc
```

---

## 2. `ActiveLearningDatasetConstruction`

The `ActiveLearningDatasetConstruction` folder contains the scripts used to run the active-learning dataset construction experiments.

### `run_al_thermo.bash`

This Bash script sequentially runs all random seeds and dataset construction strategies by calling `run_all_thermo.py`.

To run this script, open a terminal in the `ExperimentalDatasetConstruction` folder and execute:

```bash
bash run_al_thermo.bash
```

### `run_all_thermo.py`

This is the main script for running one dataset construction experiment with one random seed.

The script uses:

```text
CLEANED_DATA.csv
```

as the candidate pool.

It calls a dataset construction policy defined in:

```text
uncertainty_Thermo.py
```

The script outputs CSV files containing the performance metrics for Random Forest and XGBoost models.

The output files are saved in a folder named:

```text
al
```

Inside the `al` folder, results are organized into subfolders according to the dataset construction policy used.

### `uncertainty_Thermo.py`

This Python module contains the different dataset construction policies used in the active-learning experiments.

These policies include:

- random sampling
- query-by-committee strategies
- NSGA-II-based multi-objective strategies
- diversity-aware strategies

---

## 3. `ResultsAnalysis`

The `ResultsAnalysis` folder contains the notebooks used to analyze the active-learning results.

### `ImprovementThermoElectricData.ipynb`

This notebook plots the improvement of each dataset construction policy relative to random sampling.

Before running the notebook, make sure to update the `ROOT` path in the code so that it points to the `al` folder containing the dataset construction results.

For example:

```python
ROOT = "path/to/ExperimentalDatasetConstruction/ActiveLearningDatasetConstruction/al"
```
