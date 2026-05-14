"""Dataset configuration dataclass and loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class SourceConfig:
    """Configuration for data source."""
    type: str  # 'url', 'local', 'jarvis_api', 'custom'
    path: Optional[str] = None  # URL or file path
    format: Optional[str] = None  # 'json', 'csv', 'json.gz', etc.
    compression: Optional[str] = None  # 'gzip', None

    def __post_init__(self):
        valid_types = ['url', 'local', 'jarvis_api', 'custom']
        if self.type not in valid_types:
            raise ValueError(f"source.type must be one of {valid_types}, got '{self.type}'")

        if self.type in ['url', 'local'] and not self.path:
            raise ValueError(f"source.path is required for type '{self.type}'")


@dataclass
class FeatureColumnsConfig:
    """Configuration for feature columns."""
    type: str  # 'last_n', 'explicit', 'range'
    n: Optional[int] = None  # For 'last_n'
    columns: Optional[List[str]] = None  # For 'explicit'
    start: Optional[int] = None  # For 'range'
    end: Optional[int] = None  # For 'range'

    def __post_init__(self):
        valid_types = ['last_n', 'explicit', 'range']
        if self.type not in valid_types:
            raise ValueError(f"feature_columns.type must be one of {valid_types}, got '{self.type}'")

        if self.type == 'last_n' and self.n is None:
            raise ValueError("feature_columns.n is required when type='last_n'")

        if self.type == 'explicit' and not self.columns:
            raise ValueError("feature_columns.columns is required when type='explicit'")

        if self.type == 'range' and (self.start is None or self.end is None):
            raise ValueError("feature_columns.start and end are required when type='range'")


@dataclass
class FilterConfig:
    """Configuration for a data filter."""
    column: str
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self):
        if self.min is None and self.max is None:
            raise ValueError(f"At least one of min or max must be specified for filter on '{self.column}'")


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    drop_na_targets: bool = True
    drop_failed_structures: bool = True
    structural_feature_range: Optional[tuple] = None  # (start, end) indices for structural features
    filters: List[FilterConfig] = field(default_factory=list)
    column_renames: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlignNConfig:
    """Configuration for ALIGNN model integration."""
    enabled: bool = False
    pickle_pattern: Optional[str] = None  # e.g., 'alldataRAW_{dataset}_ALIGNN.pkl'
    jarvis_api_name: Optional[str] = None  # e.g., 'dft_3d', 'mp_3d_2020'
    id_column_in_pickle: str = 'reference'  # Column name in pickle file
    max_atoms: Optional[int] = None  # Structures with more atoms than this are screened out


@dataclass
class DatasetConfig:
    """Complete dataset configuration."""
    name: str
    source: SourceConfig
    target_columns: List[str]
    feature_columns: FeatureColumnsConfig
    id_column: str
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    alignn: AlignNConfig = field(default_factory=AlignNConfig)
    formula_column: str = 'formula'  # Column containing chemical formula

    def __post_init__(self):
        if not self.name:
            raise ValueError("Dataset name is required")

        if not self.target_columns:
            raise ValueError("At least one target column is required")

        if not self.id_column:
            raise ValueError("id_column is required")

        # Convert dicts to dataclass instances if needed
        if isinstance(self.source, dict):
            self.source = SourceConfig(**self.source)

        if isinstance(self.feature_columns, dict):
            self.feature_columns = FeatureColumnsConfig(**self.feature_columns)

        if isinstance(self.preprocessing, dict):
            # Convert filters to FilterConfig objects
            filters = self.preprocessing.get('filters', [])
            filter_objs = [FilterConfig(**f) if isinstance(f, dict) else f for f in filters]
            preprocessing_dict = {**self.preprocessing, 'filters': filter_objs}
            self.preprocessing = PreprocessingConfig(**preprocessing_dict)

        if isinstance(self.alignn, dict):
            self.alignn = AlignNConfig(**self.alignn)

    def validate(self):
        """Validate the configuration."""
        # Check that target columns and feature columns don't overlap
        if self.feature_columns.type == 'explicit':
            target_set = set(self.target_columns)
            feature_set = set(self.feature_columns.columns)
            overlap = target_set & feature_set
            if overlap:
                raise ValueError(f"Target and feature columns overlap: {overlap}")

        # Validate ALIGNN config
        if self.alignn.enabled:
            if not self.alignn.pickle_pattern:
                raise ValueError("alignn.pickle_pattern is required when ALIGNN is enabled")


def load_config(config_path: Union[str, Path]) -> DatasetConfig:
    """Load dataset configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        DatasetConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    config = DatasetConfig(**config_dict)
    config.validate()

    return config


def save_config(config: DatasetConfig, config_path: Union[str, Path]):
    """Save dataset configuration to JSON file.

    Args:
        config: DatasetConfig object to save
        config_path: Path where to save the config
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [to_dict(item) for item in obj]
        return obj

    config_dict = to_dict(config)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)
