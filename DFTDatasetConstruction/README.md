# DFT Datasets construction
Repository for DFT datasets construction; code from Rafael Espinosa Castañeda.
The code was originally developed and tested on Compute Canada / Alliance systems.

## Installation
Installation time should range from 15- 25 minutes .

### Requirements
- Python 3.11 or 3.12
- CUDA 11.8 (for GPU support)

### Installation Steps

PyTorch and DGL must be installed manually before the package because they require CUDA-specific wheels that are not on PyPI.

**1. Install PyTorch with CUDA 11.8 support:**
```bash
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

**2. Install DGL with CUDA 11.8 support:**
```bash
pip install dgl==1.1.2+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
```

**3. Install the package and remaining dependencies:**
```bash
git clone <repository-link>
cd <repository>
pip install -e .          # runtime only
pip install -e ".[dev]"   # include dev/test tools (pytest, black, mypy, etc.)
```

Tested with PyTorch 2.2.1+cu118, NumPy 1.26.4, DGL 1.1.2+cu118, torchdata 0.7.1.

## CLI Usage

Datasets are now specified via JSON config files rather than a dataset name flag. Built-in configs live in `src/multiobj_al/datasets/configs/`.

```bash
run-al --config <path/to/config.json> --growingCriteria <method>
```

### Arguments

- `--config` (**required**): Path to a dataset configuration JSON file
  - Built-in: `src/multiobj_al/datasets/configs/jarvis18.json`, `jarvis22.json`, `mp18.json`, `mp21.json`
  - Custom: any JSON file following the dataset config schema (see [DATASET_CONFIG_GUIDE.md](DATASET_CONFIG_GUIDE.md))

- `--growingCriteria` (default: `grow_MultiObj_All`): Active learning strategy
  - `random`: Random sampling baseline
  - `QBC`: Query-by-Committee
  - `grow_MultiObj_All`: Multi-objective over all three targets
  - `grow_MultiObj2Outcomes`: Multi-objective over two targets
  - `grow_QVendi`: Q-Vendi, single objective
  - `grow_QVendi2Outcomes`: Q-Vendi, two objectives
  - `grow_EA_QBC_1Obj_withDiversity_from3Outcomes`: EA+QBC, single objective with diversity
  - `grow_EA_QBC_2Obj_twoTargets_from3Outcomes`: EA+QBC, two objectives
  - `grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes`: EA+QBC, two objectives with diversity

- `--target` (default: `e_form`): Target property name(s) as defined in the config's `target_columns`
  - Single: `e_form`, `bulk_modulus`, `bandgap`
  - Paired: `eform_bulk_modulus`, `bandgap_eform`, `bandgap_bulkmodulus`

- `--outputDir` (default: `output`): Directory to save results

- `--stepFrac` (default: `0.01`): Fraction of data to add per iteration

- `--randomSeed` (default: `0`): Random seed for reproducibility

- `--alignn` (default: `False`): Use the ALIGNN model (`True`/`False`)

- `--early-stopping-patience` (default: `None`): Epochs without improvement before stopping (0 disables)

- `--early-stopping-min-delta` (default: `None`): Minimum improvement to reset the early-stopping counter

### Examples

```bash
run-al --config src/multiobj_al/datasets/configs/mp21.json \
       --growingCriteria random
```

```bash
run-al --config src/multiobj_al/datasets/configs/mp21.json \
       --growingCriteria grow_MultiObj_All \
       --outputDir results --stepFrac 0.02 --randomSeed 42
```

```bash
run-al --config src/multiobj_al/datasets/configs/jarvis18.json \
       --growingCriteria QBC --target bandgap \
       --alignn True --outputDir output_alignn
```

### Using a Custom Dataset

Create a JSON config following the schema described in [DATASET_CONFIG_GUIDE.md](DATASET_CONFIG_GUIDE.md). An annotated template is at `src/multiobj_al/datasets/configs/example_custom.json`.

```bash
run-al --config /path/to/my_dataset.json \
       --growingCriteria QBC --target my_property \
       --outputDir output
```

The config supports data from local CSV/JSON files, remote URLs, and the JARVIS API, with arbitrary numbers of targets and flexible feature column specifications.

### Output

Results are saved to `--outputDir`, including model metrics (MAE, RMSE, R²), training progress, and CSV files with predictions and uncertainties.

## Testing

```bash
pytest -v                      # fast tests only
pytest -v -m "not gpu"         # all tests, skip GPU
pytest -v -m gpu               # GPU tests (requires CUDA)
pytest --cov=src/multiobj_al   # with coverage
```

Test markers: `slow`, `gpu`

## Planned Experiments -DEMO

Random Seeds: 0, 1, 2, 3, 4

Datasets: mp21, mp18, jarvis18, jarvis22

Targets: e_form, bulk_modulus, bandgap, eform_bulk_modulus, bandgap_eform, bandgap_bulkmodulus

To test the code, we recommend to select one single random seed, a dataset and a single target. The NSGA-II policies are the most computational demanding. Running on a 
ALIENWARE-18 intel core i9 64 CPUs the time ranged from 45-60 minutes for the most demanding NSGA-II policies.

## Results Analysis

The `ResultsAnalysis` folder contains the notebooks used to analyze the active-learning results.

### `Improvement_DFTplotting.ipynb`

### `DFT_Plotting_NEW.ipynb`

These notebook plots the improvement of each dataset construction policy relative to random sampling. Also, data coverage and RMSE plots can be made for each of the different policies withe code DFT_Plotting_NEW.ipynb.

Before running the notebooks, make sure to update the `ROOT` path in the code so that it points to the `al` folder containing the dataset construction results. 
For example:

```python
ROOT = "path/to/DFTDatasetConstruction/ActiveLearningDatasetConstruction/al"
```
The results to run the analysis can be found in [Zenodo DFT ](https://zenodo.org/records/20316177?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjUwYjE0MTA0LWRjZjAtNDYzYi1iOTMzLTg1NGI0NWQ1ZWU2NiIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZGNmYzExM2MwMGMzMmNkNzQ1NTM1ZmM2ZjI5NjYwOCJ9.h4RI0JsCqdQHHkpzafBUfPA79JnnbOZwnlZDlBFyffB6E5kQq57A7XnWVE1gWQgA45VFTt88r_3atLs7hVHtRg)

