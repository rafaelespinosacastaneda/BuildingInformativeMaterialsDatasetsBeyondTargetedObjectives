# BuildingInformativeMaterialsDatasetsBeyondTargetedObjectives
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20073188.svg)](https://doi.org/10.5281/zenodo.20073188)

This repository contains the code needed to reproduce the results reported in the paper **“Building Informative Materials Datasets Beyond Targeted Objectives.”** https://arxiv.org/abs/2605.05104

The repository is organized into two main folders:

- `DFTDatasetConstruction`
- `ExperimentalDatasetConstruction`

The `DFTDatasetConstruction` folder contains the code used for the DFT dataset construction experiments. These experiments use JARVIS18, JARVIS22, MP18, and MP21 as candidate pools. This folder also includes the Zenodo link to the corresponding DFT dataset construction results.

The `ExperimentalDatasetConstruction` folder contains the code used for the experimental dataset construction experiments. These experiments use the sysTEm dataset as the candidate pool. This folder also includes the Zenodo link to the corresponding experimental dataset construction results.

The OutcomesAnalysis.ipynb code shows the properties of the properties outcome space for the MP18, MP21, JARVIS18, JARVIS22 and the curated sysTem experimental dataset.

To understand how to access the DFT or experimental results data and the corresponding code to generate the results please go to the corresponding folder.

The codes implementation make the split of datasets into the pool data to be used for dataset construction and hold out test set. If the reader is interested to visualize directly the samples used in training and test sets for each dataset, the dataset splits can be found in the [Zenodo splits ](https://zenodo.org/records/20391144?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImZlNDVhMDQ5LTZkMmEtNGY1OS1iYTE1LWVjMjJmMzI2OWJmMSIsImRhdGEiOnt9LCJyYW5kb20iOiIyM2IyMWZkOTk2NDc2NjdhMDJkZGJiZGZiYTdhOGE1ZCJ9.j4Sh_dmnF7G6sCTLOlNeSrlHISUmv5RZYxymt1WBVVIXFo7nADyPDWqzu2akCbs6X6KyXjTP_cQlnr-MkS9ilw)

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
│    ├── systemDataCuration
│    │   ├── DataCuration_sysTEmDataset.ipynb
│    │   └── sysTEm_dataset.xlsx
│    │
│    ├── ActiveLearningDatasetConstruction
│    │   ├── run_al_thermo.bash
│    │   ├── run_all_thermo.py
│    │   └── uncertainty_Thermo.py
│    │
│    └── ResultsAnalysis
│        └── ImprovementThermoElectricData.ipynb
└──OutcomesAnalysis.ipynb
```


