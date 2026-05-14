import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from multiobj_al.policies.random_policy import RandomPolicy
from multiobj_al.policies.qbc_policy import QBCPolicy
from multiobj_al.policies.multiobj_policy import MultiObjPolicy
from multiobj_al.policies.qvendi_policy import QVendiPolicy
from multiobj_al.policies.diversity_policy import DiversityPolicy


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_data():
    """Create sample training/test data for policy testing."""
    np.random.seed(0)
    x_train_val = pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
        'feature3': np.random.randn(20),
    }, index=range(20))

    y_targets_train_val = {
        'eform': pd.Series(np.random.randn(20), index=range(20)),
        'bulk_modulus': pd.Series(np.random.randn(20), index=range(20)),
        'bandgap': pd.Series(np.random.randn(20), index=range(20)),
    }

    x_test = pd.DataFrame({
        'feature1': np.random.randn(5),
        'feature2': np.random.randn(5),
        'feature3': np.random.randn(5),
    }, index=range(100, 105))

    y_targets_test = {
        'eform': pd.Series(np.random.randn(5), index=range(100, 105)),
        'bulk_modulus': pd.Series(np.random.randn(5), index=range(100, 105)),
        'bandgap': pd.Series(np.random.randn(5), index=range(100, 105)),
    }

    return {
        'x_train_val': x_train_val,
        'y_targets_train_val': y_targets_train_val,
        'x_test': x_test,
        'y_targets_test': y_targets_test,
    }


@pytest.fixture
def mock_models():
    """Create mock models that return predictions."""
    model1 = Mock()
    model2 = Mock()

    model1.fit = Mock(return_value=model1)
    model2.fit = Mock(return_value=model2)

    model1.predict = Mock(return_value=np.random.randn(5))
    model2.predict = Mock(return_value=np.random.randn(5))

    return model1, model2


@pytest.fixture
def mock_results():
    """Create mock model results for policy selection."""
    def create_results(pred_vals):
        return {
            'eform': {'model_pred': np.array(pred_vals), 'model_maes': 0.1, 'model_rmse': 0.2, 'model_r2': 0.8},
            'bulk_modulus': {'model_pred': np.array(pred_vals) * 2, 'model_maes': 0.15, 'model_rmse': 0.25, 'model_r2': 0.75},
            'bandgap': {'model_pred': np.array(pred_vals) * 0.5, 'model_maes': 0.12, 'model_rmse': 0.22, 'model_r2': 0.82},
        }
    return create_results


# ============================================================
# RandomPolicy Tests
# ============================================================

def test_random_policy_select_next_batch(sample_data, mock_models, mock_results):
    """Test RandomPolicy._select_next_batch returns random samples from the validation set."""
    model1, model2 = mock_models

    policy = RandomPolicy(
        model1=model1,
        model2=model2,
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
        int_random_seed=42,
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:]
    results_m1 = mock_results([0.1, 0.2, 0.3])
    results_m2 = mock_results([0.15, 0.25, 0.35])

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=3)

    assert len(selected) == 3
    assert all(idx in x_val.index for idx in selected)
    assert len(set(selected)) == 3  # No duplicates


def test_random_policy_deterministic_with_seed(sample_data, mock_models, mock_results):
    """Test RandomPolicy produces the same results when given the same seed."""
    model1, model2 = mock_models

    def create_policy(seed):
        return RandomPolicy(
            model1=model1,
            model2=model2,
            x_train_val=sample_data['x_train_val'],
            y_targets_train_val=sample_data['y_targets_train_val'],
            x_test=sample_data['x_test'],
            y_targets_test=sample_data['y_targets_test'],
            int_random_seed=seed,
        )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:]
    results_m1 = mock_results([0.1, 0.2, 0.3])
    results_m2 = mock_results([0.15, 0.25, 0.35])

    np.random.seed(42)
    policy1 = create_policy(42)
    selected1 = policy1._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=3)

    np.random.seed(42)
    policy2 = create_policy(42)
    selected2 = policy2._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=3)

    assert selected1 == selected2


# ============================================================
# QBCPolicy Tests
# ============================================================

def test_qbc_policy_select_highest_disagreement(sample_data, mock_models):
    """Test QBCPolicy selects the samples with highest disagreement between the two models."""
    model1, model2 = mock_models

    policy = QBCPolicy(
        model1=model1,
        model2=model2,
        target='eform',
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]  # indices 5-9

    # Disagreements: [0.1, 0.5, 0.2, 0.8, 0.1]
    # Top 2 highest: index 8 (0.8) and index 6 (0.5)
    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'model_maes': 0.1, 'model_rmse': 0.2, 'model_r2': 0.8},
        'bulk_modulus': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'model_maes': 0.1, 'model_rmse': 0.2, 'model_r2': 0.8},
        'bandgap': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'model_maes': 0.1, 'model_rmse': 0.2, 'model_r2': 0.8},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.1, 2.5, 3.2, 4.8, 5.1]), 'model_maes': 0.1, 'model_rmse': 0.2, 'model_r2': 0.8},
        'bulk_modulus': {'model_pred': np.array([1.1, 2.5, 3.2, 4.8, 5.1]), 'model_maes': 0.1, 'model_rmse': 0.2, 'model_r2': 0.8},
        'bandgap': {'model_pred': np.array([1.1, 2.5, 3.2, 4.8, 5.1]), 'model_maes': 0.1, 'model_rmse': 0.2, 'model_r2': 0.8},
    }

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2
    assert 8 in selected  # disagreement 0.8
    assert 6 in selected  # disagreement 0.5


def test_qbc_policy_handles_missing_target(sample_data, mock_models):
    """Test QBCPolicy returns empty list when the target property is not in results."""
    model1, model2 = mock_models

    policy = QBCPolicy(
        model1=model1,
        model2=model2,
        target='nonexistent_target',
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}}
    results_m2 = {'eform': {'model_pred': np.array([1.1, 2.5, 3.2, 4.8, 5.1])}}

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert selected == []


# ============================================================
# MultiObjPolicy Tests
# ============================================================

def test_multiobj_policy_two_targets(sample_data, mock_models):
    """Test MultiObjPolicy combines normalized disagreements from two target properties."""
    model1, model2 = mock_models

    policy = MultiObjPolicy(
        model1=model1,
        model2=model2,
        target1='eform',
        target2='bulk_modulus',
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        'bulk_modulus': {'model_pred': np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
        'bandgap': {'model_pred': np.array([0.1, 0.2, 0.3, 0.4, 0.5])},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])},
        'bulk_modulus': {'model_pred': np.array([15.0, 25.0, 35.0, 45.0, 55.0])},
        'bandgap': {'model_pred': np.array([0.15, 0.25, 0.35, 0.45, 0.55])},
    }

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2
    assert all(idx in x_val.index for idx in selected)


def test_multiobj_policy_single_target(sample_data, mock_models):
    """Test MultiObjPolicy uses raw absolute disagreement when given a single target."""
    model1, model2 = mock_models

    policy = MultiObjPolicy(
        model1=model1,
        model2=model2,
        target='bandgap',
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {'bandgap': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}}
    results_m2 = {'bandgap': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])}}

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2


def test_multiobj_policy_all_properties(sample_data, mock_models):
    """Test MultiObjPolicy combines disagreements across all properties when no target is set."""
    model1, model2 = mock_models

    policy = MultiObjPolicy(
        model1=model1,
        model2=model2,
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        'bulk_modulus': {'model_pred': np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
        'bandgap': {'model_pred': np.array([0.1, 0.2, 0.3, 0.4, 0.5])},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])},
        'bulk_modulus': {'model_pred': np.array([15.0, 25.0, 35.0, 45.0, 55.0])},
        'bandgap': {'model_pred': np.array([0.15, 0.25, 0.35, 0.45, 0.55])},
    }

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2


# ============================================================
# QVendiPolicy Tests
# ============================================================

def test_qvendi_policy_two_targets(sample_data, mock_models):
    """Test QVendiPolicy selection with two targets combines quality and diversity."""
    model1, model2 = mock_models

    policy = QVendiPolicy(
        model1=model1,
        model2=model2,
        target1='eform',
        target2='bulk_modulus',
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        'bulk_modulus': {'model_pred': np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])},
        'bulk_modulus': {'model_pred': np.array([15.0, 25.0, 35.0, 45.0, 55.0])},
    }

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2
    assert all(idx in x_val.index for idx in selected)


def test_qvendi_policy_single_target(sample_data, mock_models):
    """Test QVendiPolicy selection with a single target."""
    model1, model2 = mock_models

    policy = QVendiPolicy(
        model1=model1,
        model2=model2,
        target='bandgap',
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {'bandgap': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}}
    results_m2 = {'bandgap': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])}}

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2
    assert all(idx in x_val.index for idx in selected)


# ============================================================
# DiversityPolicy Tests
# ============================================================

def test_diversity_policy_two_target_with_diversity(sample_data, mock_models):
    """Test DiversityPolicy in two-target + diversity mode (NSGA-II, 3 objectives)."""
    model1, model2 = mock_models

    policy = DiversityPolicy(
        model1=model1,
        model2=model2,
        target1='eform',
        target2='bulk_modulus',
        use_diversity=True,
        pop_size=20,
        n_gen=5,
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
        int_random_seed=42,
    )

    # _ea_mode is set only when grow() is called; set it manually to test _select_next_batch directly
    assert not hasattr(policy, '_ea_mode') or policy._ea_mode is None
    policy._ea_mode = 'two_target_div'

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        'bulk_modulus': {'model_pred': np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])},
        'bulk_modulus': {'model_pred': np.array([15.0, 25.0, 35.0, 45.0, 55.0])},
    }

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2
    assert all(idx in x_val.index for idx in selected)


def test_diversity_policy_two_target_no_diversity(sample_data, mock_models):
    """Test DiversityPolicy in two-target mode without diversity (NSGA-II, 2 objectives)."""
    model1, model2 = mock_models

    policy = DiversityPolicy(
        model1=model1,
        model2=model2,
        target1='eform',
        target2='bulk_modulus',
        use_diversity=False,
        pop_size=20,
        n_gen=5,
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
        int_random_seed=42,
    )

    policy._ea_mode = 'two_target_no_div'

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        'bulk_modulus': {'model_pred': np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])},
        'bulk_modulus': {'model_pred': np.array([15.0, 25.0, 35.0, 45.0, 55.0])},
    }

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2
    assert all(idx in x_val.index for idx in selected)


def test_diversity_policy_single_target_with_diversity(sample_data, mock_models):
    """Test DiversityPolicy in single-target + diversity mode (single-objective EA)."""
    model1, model2 = mock_models

    policy = DiversityPolicy(
        model1=model1,
        model2=model2,
        target='bandgap',
        pop_size=20,
        n_gen=5,
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
        int_random_seed=42,
    )

    policy._ea_mode = 'single_target_div'

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    results_m1 = {'bandgap': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}}
    results_m2 = {'bandgap': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])}}

    selected = policy._select_next_batch(x_train, x_val, results_m1, results_m2, step_size=2)

    assert len(selected) == 2
    assert all(idx in x_val.index for idx in selected)


def test_diversity_policy_create_quality_dataframe(sample_data, mock_models):
    """Test DiversityPolicy._create_quality_dataframe builds q1/q2 columns from model disagreements."""
    model1, model2 = mock_models

    policy = DiversityPolicy(
        model1=model1,
        model2=model2,
        target1='eform',
        target2='bulk_modulus',
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        'bulk_modulus': {'model_pred': np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])},
        'bulk_modulus': {'model_pred': np.array([15.0, 25.0, 35.0, 45.0, 55.0])},
    }

    diffs_df = policy._create_quality_dataframe(results_m1, results_m2)

    assert 'q1' in diffs_df.columns
    assert 'q2' in diffs_df.columns
    assert len(diffs_df) == 5


def test_diversity_policy_scale_and_normalize(sample_data, mock_models):
    """Test DiversityPolicy._scale_and_normalize applies StandardScaler and L2 normalization."""
    model1, model2 = mock_models

    policy = DiversityPolicy(
        model1=model1,
        model2=model2,
        target='eform',
        x_train_val=sample_data['x_train_val'],
        y_targets_train_val=sample_data['y_targets_train_val'],
        x_test=sample_data['x_test'],
        y_targets_test=sample_data['y_targets_test'],
    )

    x_train = sample_data['x_train_val'].iloc[:5]
    x_val = sample_data['x_train_val'].iloc[5:10]

    X_train, X_pool = policy._scale_and_normalize(x_train, x_val, do_scale=True, normalize_feats=True)

    assert X_train.shape == (5, 3)
    assert X_pool.shape == (5, 3)
    # L2-normalized rows each have norm 1.0
    assert np.allclose(np.linalg.norm(X_train, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(X_pool, axis=1), 1.0, atol=1e-5)
