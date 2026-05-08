# BuildingInformativeMaterialsDatasetsBeyondTargetedObjectives
[![DOI](https://zenodo.org/badge/1232149192.svg)](https://doi.org/10.5281/zenodo.20073188)


This repository contains the code needed to reproduce the results reported in the paper **“Building Informative Materials Datasets Beyond Targeted Objectives.”**

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
│   └── ...
│
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
