# BuildingInformativeMaterialsDatasetsBeyondTargetedObjectives
[![DOI](https://zenodo.org/badge/1232149192.svg)](https://doi.org/10.5281/zenodo.20073188)


This repository contains the code needed to reproduce the results reported in the paper **“Building Informative Materials Datasets Beyond Targeted Objectives.”** https://arxiv.org/abs/2605.05104

The repository is organized into two main folders:

- `DFTDatasetConstruction`
- `ExperimentalDatasetConstruction`

The `DFTDatasetConstruction` folder contains the code used for the DFT dataset construction experiments. These experiments use JARVIS18, JARVIS22, MP18, and MP21 as candidate pools. This folder also includes the Zenodo link to the corresponding DFT dataset construction results.

The `ExperimentalDatasetConstruction` folder contains the code used for the experimental dataset construction experiments. These experiments use the sysTEm dataset as the candidate pool. This folder also includes the Zenodo link to the corresponding experimental dataset construction results.

---

## Repository structure

```text
.
├── DFTDatasetConstruction
│   └── src/multiobj_al
│   └── tests/
│   └── pyproject.toml
│   └── README.md
│   └── ResultsAnalysis
└── ExperimentalDatasetConstruction
    ├── systemDataCuration
    │   ├── DataCuration_sysTEmDataset.ipynb
    │   └── sysTEm_dataset.xlsx
    │
    ├── ActiveLearningDatasetConstruction
    │   ├── run_al_thermo.bash
    │   ├── run_all_thermo.py
    │   └── uncertainty_Thermo.py
    │
    └── ResultsAnalysis
        └── ImprovementThermoElectricData.ipynb 
```

## ExperimentalDatasetConstruction

The `ExperimentalDatasetConstruction` folder contains the code used to reproduce the experimental dataset construction experiments using the **sysTEm** dataset as the candidate pool.

This folder contains three subfolders:

- `systemDataCuration`
- `ActiveLearningDatasetConstruction`
- `ResultsAnalysis`

---

### 1. `systemDataCuration`

The `systemDataCuration` folder contains the code and raw data used to curate and featurize the sysTEm dataset.

#### `DataCuration_sysTEmDataset.ipynb`

This notebook featurizes the chemical compositions in `sysTEm_dataset.xlsx` and curates the dataset.

Before running the notebook, make sure that:

1. The sysTEm dataset has been downloaded.
2. The local file path to `sysTEm_dataset.xlsx` is correctly specified in the notebook.

The notebook outputs a curated CSV file named:

```text
CLEANED_DATA.csv
```

This file is used as the candidate pool for the active-learning dataset construction experiments.

#### `sysTEm_dataset.xlsx`

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

### 2. `ActiveLearningDatasetConstruction`

The `ActiveLearningDatasetConstruction` folder contains the scripts used to run the active-learning dataset construction experiments.

#### `run_al_thermo.bash`

This Bash script sequentially runs all random seeds and dataset construction strategies by calling `run_all_thermo.py`.

To run this script, open a terminal in the `ExperimentalDatasetConstruction` folder and execute:

```bash
bash run_al_thermo.bash
```

#### `run_all_thermo.py`

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

#### `uncertainty_Thermo.py`

This Python module contains the different dataset construction policies used in the active-learning experiments.

These policies include:

- random sampling
- query-by-committee strategies
- NSGA-II-based multi-objective strategies
- diversity-aware strategies

---

### 3. `ResultsAnalysis`

The `ResultsAnalysis` folder contains the notebooks used to analyze the active-learning results.

#### `ImprovementThermoElectricData.ipynb`

This notebook plots the improvement of each dataset construction policy relative to random sampling.

Before running the notebook, make sure to update the `ROOT` path in the code so that it points to the `al` folder containing the dataset construction results.

For example:

```python
ROOT = "path/to/ExperimentalDatasetConstruction/ActiveLearningDatasetConstruction/al"
```
