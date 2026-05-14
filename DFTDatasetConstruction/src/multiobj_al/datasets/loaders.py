"""Data loaders for different dataset sources."""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, config: DatasetConfig):
        """Initialize loader with dataset configuration.

        Args:
            config: Dataset configuration
        """
        self.config = config

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load dataset and return as DataFrame.

        Returns:
            DataFrame with all data from the source

        Raises:
            FileNotFoundError: If source cannot be found
            ValueError: If data format is invalid
        """

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read a file based on format and compression.

        Args:
            file_path: Path to file to read

        Returns:
            DataFrame with file contents
        """
        source = self.config.source
        format_type = source.format or file_path.suffix.lstrip('.')

        # Handle gzipped files
        if source.compression == 'gzip' or file_path.suffix == '.gz':
            if format_type == 'json' or '.json' in file_path.suffixes:
                logger.info(f"Reading gzipped JSON from {file_path}")
                return pd.read_json(file_path, compression='gzip', orient='records', lines=True)
            if format_type == 'csv':
                logger.info(f"Reading gzipped CSV from {file_path}")
                return pd.read_csv(file_path, compression='gzip')
            raise ValueError(f"Unsupported gzipped format: {format_type}")

        # Handle regular files
        if format_type == 'json':
            logger.info(f"Reading JSON from {file_path}")
            return pd.read_json(file_path, orient='records', lines=True)
        if format_type == 'csv':
            logger.info(f"Reading CSV from {file_path}")
            return pd.read_csv(file_path)
        if format_type in ['pkl', 'pickle']:
            logger.info(f"Reading pickle from {file_path}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        raise ValueError(f"Unsupported file format: {format_type}")


class URLLoader(DatasetLoader):
    """Loader for datasets from HTTP/HTTPS URLs."""

    def load(self) -> pd.DataFrame:
        """Download and load dataset from URL.

        Returns:
            DataFrame with dataset contents
        """
        url = self.config.source.path
        if not url:
            raise ValueError("URL path is required for URLLoader")

        # Determine local filename from URL
        filename = url.split('/')[-1].split('?')[0]  # Remove query params
        local_path = Path('./data') / filename

        # Create data directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download if not already cached
        if not local_path.exists():
            logger.info(f"Downloading {url} to {local_path}")
            urlretrieve(url, local_path)
            logger.info("Download complete")
        else:
            logger.info(f"Using cached file: {local_path}")

        return self._read_file(local_path)


class LocalFileLoader(DatasetLoader):
    """Loader for local files (CSV, JSON, pickle)."""

    def load(self) -> pd.DataFrame:
        """Load dataset from local file.

        Returns:
            DataFrame with dataset contents
        """
        file_path = Path(self.config.source.path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        return self._read_file(file_path)


class JarvisAPILoader(DatasetLoader):
    """Loader for JARVIS datasets via figshare API."""

    def load(self) -> pd.DataFrame:
        """Load dataset from JARVIS figshare API.

        Returns:
            DataFrame with dataset contents
        """
        try:
            from jarvis.db.figshare import data as jarvis_figshare_data
        except ImportError as exc:
            raise ImportError(
                "jarvis-tools is required for JarvisAPILoader. "
                "Install it with: pip install jarvis-tools"
            ) from exc

        if not self.config.alignn.jarvis_api_name:
            raise ValueError("jarvis_api_name is required in alignn config for JarvisAPILoader")

        # Check for cached pickle file
        pickle_pattern = self.config.alignn.pickle_pattern or f"alldataRAW_{self.config.name}_ALIGNN.pkl"
        local_path = Path(pickle_pattern)

        if not local_path.exists():
            logger.info(f"Downloading {self.config.alignn.jarvis_api_name} from JARVIS figshare")
            data = jarvis_figshare_data(self.config.alignn.jarvis_api_name)
            with open(local_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved to {local_path}")
        else:
            logger.info(f"Using cached JARVIS data: {local_path}")
            with open(local_path, 'rb') as f:
                data = pickle.load(f)

        return pd.DataFrame(data)


class CustomLoader(DatasetLoader):
    """Loader that uses a custom user-provided loading function."""

    def __init__(self, config: DatasetConfig, load_fn):
        """Initialize with config and custom loading function.

        Args:
            config: Dataset configuration
            load_fn: Callable that returns a DataFrame
        """
        super().__init__(config)
        self.load_fn = load_fn

    def load(self) -> pd.DataFrame:
        """Load dataset using custom function.

        Returns:
            DataFrame from custom loading function
        """
        logger.info("Loading dataset using custom function")
        df = self.load_fn(self.config)

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Custom loader must return a pandas DataFrame")

        return df


def get_loader(config: DatasetConfig, custom_load_fn=None) -> DatasetLoader:
    """Factory function to get appropriate loader for config.

    Args:
        config: Dataset configuration
        custom_load_fn: Optional custom loading function for 'custom' type

    Returns:
        Appropriate DatasetLoader instance

    Raises:
        ValueError: If source type is unknown or custom_load_fn not provided for 'custom' type
    """
    source_type = config.source.type

    if source_type == 'url':
        return URLLoader(config)
    if source_type == 'local':
        return LocalFileLoader(config)
    if source_type == 'jarvis_api':
        return JarvisAPILoader(config)
    if source_type == 'custom':
        if custom_load_fn is None:
            raise ValueError("custom_load_fn must be provided for 'custom' source type")
        return CustomLoader(config, custom_load_fn)
    raise ValueError(f"Unknown source type: {source_type}")
