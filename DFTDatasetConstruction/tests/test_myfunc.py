"""Tests for myfunc.get_scores."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from multiobj_al.myfunc import get_scores


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(0)
    idx_train = list(range(15))
    idx_test = list(range(15, 20))
    X_train = pd.DataFrame(rng.standard_normal((15, 5)), index=idx_train)
    y_train = pd.Series(rng.standard_normal(15), index=idx_train)
    X_test = pd.DataFrame(rng.standard_normal((5, 5)), index=idx_test)
    y_test = pd.Series(rng.standard_normal(5), index=idx_test)
    return X_train, y_train, X_test, y_test


def test_get_scores_basic(small_dataset):
    X_train, y_train, X_test, y_test = small_dataset
    result = get_scores(LinearRegression(), X_train, y_train, X_test, y_test)
    assert 'model_maes' in result
    assert 'model_rmse' in result
    assert 'model_r2' in result
    assert not np.isnan(result['model_maes'])


def test_get_scores_alignn_missing_dataset(small_dataset, tmp_path, monkeypatch):
    """When an ALIGNN model is provided but no data file exists, ALIGNN metrics are NaN
    and normal ML metrics are still computed correctly."""
    X_train, y_train, X_test, y_test = small_dataset

    # Run from tmp_path so there is no stray pkl file present
    monkeypatch.chdir(tmp_path)

    sample_ids = pd.Series(
        [f'id_{i}' for i in X_train.index],
        index=X_train.index,
    )

    result = get_scores(
        model=LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        alignn_model=LinearRegression(),   # not None — triggers the ALIGNN code path
        dataset='nonexistent_dataset',     # no pkl file for this name
        sample_ids=sample_ids,
    )

    # ALIGNN metrics must be NaN because the data file is absent
    assert np.isnan(result['alignn_maes'])
    assert np.isnan(result['alignn_rmse'])
    assert np.isnan(result['alignn_r2'])

    # Normal ML metrics must still be computed
    assert not np.isnan(result['model_maes'])
    assert not np.isnan(result['model_rmse'])


def test_get_scores_alignn_no_sample_ids(small_dataset, monkeypatch):
    """When an ALIGNN model is provided but sample_ids is None, ALIGNN metrics are NaN."""
    X_train, y_train, X_test, y_test = small_dataset

    result = get_scores(
        model=LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        alignn_model=LinearRegression(),
        dataset='some_dataset',
        sample_ids=None,
    )

    assert np.isnan(result['alignn_maes'])
    assert np.isnan(result['alignn_rmse'])
    assert np.isnan(result['alignn_r2'])
