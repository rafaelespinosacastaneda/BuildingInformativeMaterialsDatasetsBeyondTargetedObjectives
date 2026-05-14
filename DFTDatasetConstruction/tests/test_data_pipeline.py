"""Tests for data pipeline utilities: drop_failed_structures, return_model,
get_loader, LocalFileLoader format dispatch, DatasetConfig.validate,
and load_config / save_config roundtrip."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from multiobj_al.datasets.dataset_config import (
    DatasetConfig,
    load_config,
    save_config,
)
from multiobj_al.datasets.loaders import (
    CustomLoader,
    JarvisAPILoader,
    LocalFileLoader,
    URLLoader,
    get_loader,
)
from multiobj_al.distill import return_model
from multiobj_al.myfunc import drop_failed_structures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(source_type='local', path='/tmp/fake.csv', **overrides):
    """Build a minimal DatasetConfig for testing."""
    kwargs = dict(
        name='test_dataset',
        source={'type': source_type, 'path': path},
        target_columns=['target'],
        feature_columns={'type': 'last_n', 'n': 2},
        id_column='id',
    )
    kwargs.update(overrides)
    return DatasetConfig(**kwargs)


# ---------------------------------------------------------------------------
# drop_failed_structures
# ---------------------------------------------------------------------------

class TestDropFailedStructures:

    def test_explicit_labels_drops_nan_rows(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.standard_normal((10, 5)), columns=list('abcde'))
        df.loc[2, 'b'] = np.nan
        df.loc[7, 'b'] = np.nan

        result = drop_failed_structures(df, label_stuc=['b'])

        assert len(result) == 8
        assert 2 not in result.index
        assert 7 not in result.index

    def test_explicit_labels_no_nans_keeps_all(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.standard_normal((6, 4)), columns=list('abcd'))

        result = drop_failed_structures(df, label_stuc=['a', 'b'])

        assert len(result) == 6

    def test_default_range_on_small_df_keeps_all(self):
        # With fewer than 273 columns the default slice [-273:-145] is empty,
        # so no NaN detection fires and every row is retained.
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.standard_normal((5, 10)))
        df.iloc[1, 3] = np.nan  # NaN in col 3 — outside the (empty) default range

        result = drop_failed_structures(df)

        assert len(result) == 5


# ---------------------------------------------------------------------------
# return_model
# ---------------------------------------------------------------------------

class TestReturnModel:

    def test_xgb_returns_xgbregressor(self):
        model = return_model('xgb', random_state=0)
        assert isinstance(model, xgb.XGBRegressor)

    def test_rf_returns_random_forest(self):
        model = return_model('rf', random_state=42)
        assert isinstance(model, RandomForestRegressor)

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match='Unknown model'):
            return_model('bad_model', random_state=0)


# ---------------------------------------------------------------------------
# get_loader
# ---------------------------------------------------------------------------

class TestGetLoader:

    def test_local_source_returns_local_file_loader(self):
        config = _minimal_config(source_type='local', path='/tmp/fake.csv')
        assert isinstance(get_loader(config), LocalFileLoader)

    def test_url_source_returns_url_loader(self):
        config = _minimal_config(source_type='url', path='http://example.com/data.csv')
        assert isinstance(get_loader(config), URLLoader)

    def test_custom_without_fn_raises(self):
        config = _minimal_config(source_type='custom', path=None)
        with pytest.raises(ValueError, match='custom_load_fn'):
            get_loader(config)

    def test_custom_with_fn_returns_custom_loader(self):
        config = _minimal_config(source_type='custom', path=None)
        loader = get_loader(config, custom_load_fn=lambda c: pd.DataFrame())
        assert isinstance(loader, CustomLoader)

    def test_unknown_source_type_raises(self):
        config = _minimal_config(source_type='local', path='/tmp/fake.csv')
        config.source.type = 'totally_unknown'  # bypass __post_init__ validation
        with pytest.raises(ValueError, match='Unknown source type'):
            get_loader(config)


# ---------------------------------------------------------------------------
# LocalFileLoader — format dispatch via _read_file
# ---------------------------------------------------------------------------

class TestLocalFileLoaderFormats:

    def test_csv(self, tmp_path):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        p = tmp_path / 'data.csv'
        df.to_csv(p, index=False)

        config = _minimal_config(source_type='local', path=str(p))
        result = LocalFileLoader(config).load()

        assert list(result.columns) == ['a', 'b']
        assert len(result) == 3

    def test_json_lines(self, tmp_path):
        df = pd.DataFrame({'x': [10, 20], 'y': [30, 40]})
        p = tmp_path / 'data.json'
        df.to_json(p, orient='records', lines=True)

        config = _minimal_config(source_type='local', path=str(p))
        result = LocalFileLoader(config).load()

        assert list(result.columns) == ['x', 'y']
        assert len(result) == 2

    def test_pickle_dataframe(self, tmp_path):
        df = pd.DataFrame({'m': [7, 8], 'n': [9, 10]})
        p = tmp_path / 'data.pkl'
        with open(p, 'wb') as f:
            pickle.dump(df, f)

        config = _minimal_config(source_type='local', path=str(p))
        result = LocalFileLoader(config).load()

        assert list(result.columns) == ['m', 'n']
        assert len(result) == 2

    def test_pickle_list_of_dicts(self, tmp_path):
        records = [{'p': 1, 'q': 2}, {'p': 3, 'q': 4}]
        p = tmp_path / 'data.pkl'
        with open(p, 'wb') as f:
            pickle.dump(records, f)

        config = _minimal_config(source_type='local', path=str(p))
        result = LocalFileLoader(config).load()

        assert set(result.columns) == {'p', 'q'}
        assert len(result) == 2

    def test_missing_file_raises(self, tmp_path):
        config = _minimal_config(source_type='local', path=str(tmp_path / 'no_such_file.csv'))
        with pytest.raises(FileNotFoundError):
            LocalFileLoader(config).load()


# ---------------------------------------------------------------------------
# DatasetConfig.validate
# ---------------------------------------------------------------------------

class TestDatasetConfigValidate:

    def test_explicit_feature_target_overlap_raises(self):
        config = DatasetConfig(
            name='overlap_test',
            source={'type': 'local', 'path': '/tmp/fake.csv'},
            target_columns=['target'],
            feature_columns={'type': 'explicit', 'columns': ['feat1', 'target']},
            id_column='id',
        )
        with pytest.raises(ValueError, match='overlap'):
            config.validate()

    def test_no_overlap_passes(self):
        config = DatasetConfig(
            name='no_overlap',
            source={'type': 'local', 'path': '/tmp/fake.csv'},
            target_columns=['target'],
            feature_columns={'type': 'explicit', 'columns': ['feat1', 'feat2']},
            id_column='id',
        )
        config.validate()  # must not raise

    def test_alignn_enabled_without_pickle_pattern_raises(self):
        config = DatasetConfig(
            name='alignn_test',
            source={'type': 'local', 'path': '/tmp/fake.csv'},
            target_columns=['target'],
            feature_columns={'type': 'last_n', 'n': 3},
            id_column='id',
            alignn={'enabled': True, 'pickle_pattern': None},
        )
        with pytest.raises(ValueError, match='pickle_pattern'):
            config.validate()

    def test_alignn_enabled_with_pickle_pattern_passes(self):
        config = DatasetConfig(
            name='alignn_ok',
            source={'type': 'local', 'path': '/tmp/fake.csv'},
            target_columns=['target'],
            feature_columns={'type': 'last_n', 'n': 3},
            id_column='id',
            alignn={'enabled': True, 'pickle_pattern': 'alldataRAW_test_ALIGNN.pkl'},
        )
        config.validate()  # must not raise


# ---------------------------------------------------------------------------
# load_config / save_config
# ---------------------------------------------------------------------------

class TestLoadSaveConfig:

    def test_load_config_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / 'nonexistent.json')

    def test_roundtrip(self, tmp_path):
        config = DatasetConfig(
            name='roundtrip',
            source={'type': 'local', 'path': '/tmp/fake.csv'},
            target_columns=['prop1', 'prop2'],
            feature_columns={'type': 'last_n', 'n': 7},
            id_column='sample_id',
        )
        config_path = tmp_path / 'config.json'
        save_config(config, config_path)

        loaded = load_config(config_path)

        assert loaded.name == 'roundtrip'
        assert loaded.target_columns == ['prop1', 'prop2']
        assert loaded.feature_columns.n == 7
        assert loaded.source.type == 'local'
        assert loaded.id_column == 'sample_id'

    def test_save_creates_parent_dirs(self, tmp_path):
        config = DatasetConfig(
            name='nested',
            source={'type': 'local', 'path': '/tmp/fake.csv'},
            target_columns=['t'],
            feature_columns={'type': 'last_n', 'n': 1},
            id_column='id',
        )
        config_path = tmp_path / 'deep' / 'nested' / 'config.json'
        save_config(config, config_path)

        assert config_path.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        config = DatasetConfig(
            name='json_check',
            source={'type': 'local', 'path': '/tmp/fake.csv'},
            target_columns=['t'],
            feature_columns={'type': 'last_n', 'n': 1},
            id_column='id',
        )
        config_path = tmp_path / 'config.json'
        save_config(config, config_path)

        with open(config_path, encoding='utf-8') as f:
            data = json.load(f)

        assert data['name'] == 'json_check'
