# Dataset Configuration Guide

## Overview

The active learning codebase supports flexible dataset configuration through JSON files. This allows you to easily add custom datasets without modifying the core code.

## Quick Start

### Using Built-in Datasets

Run active learning with one of the pre-configured datasets:

```bash
python -m src.multiobj_al.run_al \
    --config src/multiobj_al/datasets/configs/jarvis18.json \
    --growingCriteria QBC \
    --target bandgap \
    --outputDir output \
    --randomSeed 0
```

### Available Built-in Datasets

- `jarvis18.json` - JARVIS 2018 dataset (DFT 3D materials)
- `jarvis22.json` - JARVIS 2022 dataset (DFT 3D 2021)
- `mp18.json` - Materials Project 2018
- `mp21.json` - Materials Project 2021

## Dataset Configuration Format

A dataset configuration file is a JSON file that specifies:

1. **Data source** (URL, local file, or API)
2. **Target properties** to predict
3. **Feature columns** specification
4. **Preprocessing** steps and filters
5. **ALIGNN integration** (optional)

### Example Configuration

```json
{
  "name": "my_dataset",
  "source": {
    "type": "local",
    "path": "/path/to/data.csv",
    "format": "csv"
  },
  "target_columns": ["property1", "property2"],
  "feature_columns": {
    "type": "explicit",
    "columns": ["feat1", "feat2", "feat3"]
  },
  "id_column": "material_id",
  "formula_column": "formula",
  "preprocessing": {
    "drop_na_targets": true,
    "drop_failed_structures": false,
    "filters": [
      {
        "column": "property1",
        "min": 0,
        "max": 100
      }
    ],
    "column_renames": {}
  },
  "alignn": {
    "enabled": false
  }
}
```

## Configuration Fields

### Required Fields

#### `name` (string)
Identifier for the dataset.

#### `source` (object)
Specifies where to load the data from.

**Fields:**
- `type`: One of `"url"`, `"local"`, `"jarvis_api"`, or `"custom"`
- `path`: URL or file path (required for `url` and `local` types)
- `format`: Data format - `"json"`, `"csv"`, etc.
- `compression`: Optional compression - `"gzip"` or `null`

**Examples:**

```json
// Load from URL
"source": {
  "type": "url",
  "path": "https://example.com/data.json.gz",
  "format": "json",
  "compression": "gzip"
}

// Load from local file
"source": {
  "type": "local",
  "path": "./data/my_dataset.csv",
  "format": "csv"
}

// Load from JARVIS API
"source": {
  "type": "jarvis_api"
}
```

#### `target_columns` (array of strings)
List of column names for target properties to predict.

```json
"target_columns": ["e_form", "bulk_modulus", "bandgap"]
```

#### `feature_columns` (object)
Specification for which columns contain ML features.

**Types:**

1. **Last N columns:**
   ```json
   "feature_columns": {
     "type": "last_n",
     "n": 273
   }
   ```

2. **Explicit list:**
   ```json
   "feature_columns": {
     "type": "explicit",
     "columns": ["feature1", "feature2", "feature3"]
   }
   ```

3. **Column range:**
   ```json
   "feature_columns": {
     "type": "range",
     "start": 10,
     "end": 283
   }
   ```

#### `id_column` (string)
Column name containing unique material/sample identifiers.

```json
"id_column": "material_id"
```

### Optional Fields

#### `formula_column` (string, default: `"formula"`)
Column containing chemical formulas.

#### `preprocessing` (object)
Data preprocessing configuration.

**Fields:**
- `drop_na_targets` (boolean): Drop rows with NaN in target columns
- `drop_failed_structures` (boolean): Drop rows with failed structural features
- `structural_feature_range` (array): `[start_index, end_index]` for structural features
- `filters` (array): Quality filters to apply
- `column_renames` (object): Column name mappings

**Example:**
```json
"preprocessing": {
  "drop_na_targets": true,
  "drop_failed_structures": true,
  "structural_feature_range": [-273, -145],
  "filters": [
    {"column": "e_form", "max": 5.0},
    {"column": "bulk_modulus", "min": 0.0}
  ],
  "column_renames": {
    "formula_pretty": "formula"
  }
}
```

#### `alignn` (object)
ALIGNN model integration configuration.

**Fields:**
- `enabled` (boolean): Whether ALIGNN is supported for this dataset
- `pickle_pattern` (string): Filename pattern for ALIGNN data
- `jarvis_api_name` (string): JARVIS API dataset name
- `id_column_in_pickle` (string): ID column name in pickle file

```json
"alignn": {
  "enabled": false,
  "pickle_pattern": "alldataRAW_my_dataset_ALIGNN.pkl",
  "jarvis_api_name": "dft_3d",
  "id_column_in_pickle": "reference"
}
```

## Creating a Custom Dataset

### Step 1: Prepare Your Data

Your data should be in CSV or JSON format with:
- Feature columns (numeric)
- Target property columns (numeric)
- ID column (string or numeric)
- Optional: Chemical formula column

### Step 2: Create Configuration File

Create a JSON file (e.g., `my_dataset.json`) with your dataset specification:

```json
{
  "name": "my_dataset",
  "source": {
    "type": "local",
    "path": "./data/my_data.csv",
    "format": "csv"
  },
  "target_columns": ["target_prop1", "target_prop2"],
  "feature_columns": {
    "type": "explicit",
    "columns": ["feature1", "feature2", "feature3", "feature4"]
  },
  "id_column": "sample_id",
  "preprocessing": {
    "drop_na_targets": true,
    "filters": [
      {"column": "target_prop1", "min": 0}
    ]
  },
  "alignn": {
    "enabled": false
  }
}
```

### Step 3: Run Active Learning

```bash
python -m src.multiobj_al.run_al \
    --config my_dataset.json \
    --growingCriteria random \
    --target target_prop1 \
    --outputDir results \
    --randomSeed 0
```

## Active Learning Criteria

### Single-Target Criteria

These require specifying one target property:

- `random`: Random sampling
- `QBC`: Query-by-Committee (model disagreement)
- `grow_QVendi`: Q-Vendi score (quality + diversity)
- `grow_EA_QBC_1Obj_withDiversity_from3Outcomes`: Evolutionary algorithm with diversity

**Example:**
```bash
python -m src.multiobj_al.run_al \
    --config jarvis18.json \
    --growingCriteria QBC \
    --target bandgap \
    --outputDir output
```

### Multi-Target Criteria

These work with all available targets:

- `grow_MultiObj_All`: Multi-objective combining all targets

### Pair-Target Criteria

These require specifying two targets (e.g., `e_form_bulk_modulus`):

- `grow_MultiObj2Outcomes`: Two-target multi-objective
- `grow_QVendi2Outcomes`: Q-Vendi with two targets
- `grow_EA_QBC_2Obj_twoTargets_from3Outcomes`: EA with two QBC objectives
- `grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes`: EA with two QBC + diversity

**Example:**
```bash
python -m src.multiobj_al.run_al \
    --config jarvis18.json \
    --growingCriteria grow_MultiObj2Outcomes \
    --target e_form_bulk_modulus \
    --outputDir output
```

## Common Use Cases

### 1. Different Materials Database

Use data from OQMD, AFLOW, or other databases:

```json
{
  "name": "oqmd_subset",
  "source": {
    "type": "local",
    "path": "./data/oqmd_data.csv",
    "format": "csv"
  },
  "target_columns": ["formation_energy", "band_gap"],
  "feature_columns": {
    "type": "last_n",
    "n": 145
  },
  "id_column": "icsd_id",
  "formula_column": "composition"
}
```

### 2. Custom ML Features

Use custom features instead of Matminer:

```json
{
  "name": "custom_features",
  "source": {
    "type": "local",
    "path": "./data/custom_featurized.csv",
    "format": "csv"
  },
  "target_columns": ["property_of_interest"],
  "feature_columns": {
    "type": "explicit",
    "columns": ["custom_feat_1", "custom_feat_2", "custom_feat_3"]
  },
  "id_column": "material_id"
}
```

### 3. Different Property Types

Target different material properties:

```json
{
  "name": "thermal_properties",
  "source": {
    "type": "local",
    "path": "./data/thermal_data.csv",
    "format": "csv"
  },
  "target_columns": [
    "thermal_conductivity",
    "seebeck_coefficient",
    "electrical_conductivity"
  ],
  "feature_columns": {
    "type": "last_n",
    "n": 273
  },
  "id_column": "material_id"
}
```

## Troubleshooting

### Configuration Validation Errors

If you get validation errors, check:
- All required fields are present
- `source.type` is one of: `url`, `local`, `jarvis_api`, `custom`
- `feature_columns.type` is one of: `last_n`, `explicit`, `range`
- Target columns and feature columns don't overlap

### Data Loading Errors

If data fails to load:
- Verify file path is correct
- Check file format matches `source.format`
- Ensure compression setting matches actual file
- Verify column names in config match actual data

### Missing Features

If you get "column not found" errors:
- Check that feature column names match the data
- Verify target column names exist
- Ensure ID column name is correct

## Advanced Topics

### Programmatic Dataset Creation

You can create dataset configs programmatically:

```python
from src.multiobj_al.datasets import DatasetConfig, save_config

config = DatasetConfig(
    name="my_dataset",
    source={"type": "local", "path": "./data.csv", "format": "csv"},
    target_columns=["prop1", "prop2"],
    feature_columns={"type": "last_n", "n": 100},
    id_column="id"
)

save_config(config, "my_dataset.json")
```

### Custom Data Loaders

For complex data sources, you can create a custom loader:

```python
from src.multiobj_al.datasets import load_config

config = load_config("my_dataset.json")

# Modify config.source to use custom loader
config.source.type = "custom"

# Pass custom loading function to get_data
def my_custom_loader(config):
    # Your custom data loading logic
    return dataframe

# Use in active learning workflow
```
