import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from multiobj_al.policies.active_learning_grower import (
    ActiveLearningGrower,
    entropy_q,
    vendi_scorefast,
    q_vendi_score,
    q_vendi_scorefast,
    Bestindices_qVendiScore,
    Bestindices_qVendiScoreFast,
    Bestindices_qVendiScoreBatch,
)


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_grower_data():
    """Create sample data for ActiveLearningGrower tests."""
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


class ConcreteGrower(ActiveLearningGrower):
    """Concrete implementation of ActiveLearningGrower for testing."""

    def __init__(self, model1=None, model2=None, **kwargs):
        super().__init__(**kwargs)
        self.model1 = model1
        self.model2 = model2

    def grow(self):
        return self._run_al_loop()

    def _select_next_batch(self, x_train, x_val, results_model1, results_model2, step_size):
        return np.random.choice(x_val.index, min(step_size, len(x_val)), replace=False).tolist()


# ============================================================
# ActiveLearningGrower Initialization Tests
# ============================================================

def test_active_learning_grower_initialization(sample_grower_data):
    """Test ActiveLearningGrower initializes with correct parameters."""
    grower = ConcreteGrower(**sample_grower_data)

    assert grower.x_train_val.equals(sample_grower_data['x_train_val'])
    assert grower.y_targets_train_val == sample_grower_data['y_targets_train_val']
    assert grower.int_random_seed == 0  # Default value


def test_active_learning_grower_custom_parameters(sample_grower_data):
    """Test ActiveLearningGrower accepts custom parameters."""
    grower = ConcreteGrower(
        **sample_grower_data,
        str_save_dir='custom_dir',
        int_random_seed=42,
        flt_step_frac=0.05,
        n_round=8,
    )

    assert grower.str_save_dir == 'custom_dir'
    assert grower.int_random_seed == 42
    assert grower.flt_step_frac == 0.05
    assert grower.n_round == 8


# ============================================================
# Helper Method Tests
# ============================================================

def test_prepare_y_trainval(sample_grower_data):
    """Test _prepare_y_trainval combines target series correctly."""
    grower = ConcreteGrower(**sample_grower_data)

    y_trainval = grower._prepare_y_trainval()

    assert isinstance(y_trainval, pd.DataFrame)
    assert 'eformTrainVal' in y_trainval.columns
    assert 'bulk_modulusTrainVal' in y_trainval.columns
    assert 'bandgapTrainVal' in y_trainval.columns
    assert len(y_trainval) == len(sample_grower_data['y_targets_train_val']['eform'])


def test_initialize_al_loop(sample_grower_data):
    """Test _initialize_al_loop sets up AL parameters correctly."""
    grower = ConcreteGrower(**sample_grower_data, flt_step_frac=0.1, int_random_seed=42)

    initial_train_size, step_size, initial_indices = grower._initialize_al_loop()

    expected_size = int(np.rint(len(sample_grower_data['x_train_val']) * 0.1))
    assert initial_train_size == expected_size
    assert step_size == expected_size
    assert len(initial_indices) == expected_size
    assert all(idx in sample_grower_data['x_train_val'].index for idx in initial_indices)


def test_split_train_val(sample_grower_data):
    """Test _split_train_val correctly splits data."""
    grower = ConcreteGrower(**sample_grower_data)

    y_trainval = grower._prepare_y_trainval()
    train_idx = [0, 1, 2, 3, 4]

    x_train, x_val, y_train, y_val = grower._split_train_val(
        sample_grower_data['x_train_val'],
        y_trainval,
        train_idx,
    )

    assert len(x_train) == 5
    assert len(x_val) == 15
    assert 'eformTrain' in y_train.columns
    assert 'eformVal' in y_val.columns
    assert set(x_train.index) == set(train_idx)


def test_min_max_standardize(sample_grower_data):
    """Test _min_max_standardize normalizes data to [0,1]."""
    grower = ConcreteGrower(**sample_grower_data)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = grower._min_max_standardize(data)

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert np.allclose(result, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_min_max_standardize_handles_constant(sample_grower_data):
    """Test _min_max_standardize handles constant arrays."""
    grower = ConcreteGrower(**sample_grower_data)

    data = np.array([5.0, 5.0, 5.0])
    result = grower._min_max_standardize(data)

    assert np.allclose(result, [0.0, 0.0, 0.0])


def test_combine_normalized_differences(sample_grower_data):
    """Test _combine_normalized_differences combines target disagreements."""
    grower = ConcreteGrower(**sample_grower_data)

    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        'bulk_modulus': {'model_pred': np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.5, 2.5, 3.5, 4.5, 5.5])},
        'bulk_modulus': {'model_pred': np.array([15.0, 25.0, 35.0, 45.0, 55.0])},
    }

    combined = grower._combine_normalized_differences(results_m1, results_m2, 'eform', 'bulk_modulus')

    assert len(combined) == 5
    assert combined.min() >= 0.0


def test_combine_all_properties_differences(sample_grower_data):
    """Test _combine_all_properties_differences sums normalized disagreements across all properties."""
    grower = ConcreteGrower(**sample_grower_data)

    results_m1 = {
        'eform': {'model_pred': np.array([1.0, 2.0, 3.0])},
        'bulk_modulus': {'model_pred': np.array([10.0, 20.0, 30.0])},
        'bandgap': {'model_pred': np.array([0.1, 0.2, 0.3])},
    }
    results_m2 = {
        'eform': {'model_pred': np.array([1.5, 2.5, 3.5])},
        'bulk_modulus': {'model_pred': np.array([15.0, 25.0, 35.0])},
        'bandgap': {'model_pred': np.array([0.15, 0.25, 0.35])},
    }

    combined = grower._combine_all_properties_differences(results_m1, results_m2)

    assert len(combined) == 3
    assert combined.min() >= 0.0


def test_calculate_multi_property_scores(sample_grower_data):
    """Test _calculate_multi_property_scores fits a model and returns metrics for each property."""
    from sklearn.linear_model import LinearRegression

    grower = ConcreteGrower(**sample_grower_data)

    y_trainval = grower._prepare_y_trainval()
    train_idx = list(range(10))
    x_train, x_val, y_train_dict, _ = grower._split_train_val(
        sample_grower_data['x_train_val'], y_trainval, train_idx
    )

    results = grower._calculate_multi_property_scores(
        LinearRegression(),
        x_train,
        y_train_dict,
        sample_grower_data['x_test'],
        sample_grower_data['y_targets_test'],
        x_val,
    )

    assert set(results.keys()) == {'eform', 'bulk_modulus', 'bandgap'}
    for prop in results:
        assert 'model_maes' in results[prop]
        assert 'model_rmse' in results[prop]
        assert 'model_r2' in results[prop]
        assert 'model_pred' in results[prop]
        assert len(results[prop]['model_pred']) == len(x_val)


# ============================================================
# NSGA-II Algorithm Tests
# ============================================================

def test_fast_non_dominated_sort(sample_grower_data):
    """Test _fast_non_dominated_sort identifies Pareto fronts."""
    grower = ConcreteGrower(**sample_grower_data)

    # Individual 0: [3, 3] - dominates all others
    # Individual 1: [2, 2] - dominated by 0
    # Individual 2: [1, 3] - on Pareto front with 0
    # Individual 3: [3, 1] - on Pareto front with 0
    F = np.array([
        [3, 3],
        [2, 2],
        [1, 3],
        [3, 1],
    ])

    fronts, rank = grower._fast_non_dominated_sort(F)

    assert len(fronts) > 0
    assert 0 in fronts[0]  # Individual 0 should be in first front
    assert all(rank[i] == 0 for i in fronts[0])


def test_crowding_distance(sample_grower_data):
    """Test _crowding_distance gives boundary points infinite distance."""
    grower = ConcreteGrower(**sample_grower_data)

    F = np.array([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0],
    ])
    front = [0, 1, 2]

    crowd = grower._crowding_distance(F, front)

    assert crowd[0] == np.inf
    assert crowd[2] == np.inf
    assert np.isfinite(crowd[1])


def test_crowding_distance_small_front(sample_grower_data):
    """Test _crowding_distance gives infinite distance when front has 2 or fewer members."""
    grower = ConcreteGrower(**sample_grower_data)

    F = np.array([[1.0, 1.0], [2.0, 2.0]])
    front = [0, 1]

    crowd = grower._crowding_distance(F, front)

    assert all(crowd[i] == np.inf for i in front)


def test_make_unique(sample_grower_data):
    """Test _make_unique removes duplicate genes from a chromosome."""
    grower = ConcreteGrower(**sample_grower_data)
    rng = np.random.default_rng(42)

    chrom = np.array([1, 2, 2, 3, 4, 4, 5])
    pool_size = 10

    result = grower._make_unique(chrom, pool_size, rng)

    assert len(set(result)) == len(result)
    assert all(0 <= g < pool_size for g in result)


def test_ordered_crossover(sample_grower_data):
    """Test _ordered_crossover produces offspring of the correct length."""
    grower = ConcreteGrower(**sample_grower_data)
    rng = np.random.default_rng(42)

    parent1 = np.array([0, 1, 2, 3, 4])
    parent2 = np.array([4, 3, 2, 1, 0])

    child1, child2 = grower._ordered_crossover(parent1, parent2, rng)

    assert len(child1) == len(parent1)
    assert len(child2) == len(parent2)


def test_mutate_swap(sample_grower_data):
    """Test _mutate_swap preserves chromosome length."""
    grower = ConcreteGrower(**sample_grower_data)
    rng = np.random.default_rng(42)

    chrom = np.array([0, 1, 2, 3, 4])
    pool_size = 10

    result = grower._mutate_swap(chrom, pool_size, rng, p_mut=1.0)

    assert len(result) == len(chrom)


def test_nsga2_evolve(sample_grower_data):
    """Test _nsga2_evolve returns a valid chromosome of the right size."""
    grower = ConcreteGrower(**sample_grower_data)

    n_pool = 15
    step_size = 3

    def fitness(chrom):
        idx = np.asarray(chrom, dtype=int)
        # Two competing objectives so NSGA-II has something meaningful to solve
        return float(np.mean(idx)), float(n_pool - 1 - np.mean(idx))

    result = grower._nsga2_evolve(
        fitness, n_pool=n_pool, step_size=step_size,
        pop_size=10, n_gen=5, p_cx=0.9, p_mut=0.25, random_state=42,
    )

    assert result is not None
    assert len(result) == step_size
    assert all(0 <= g < n_pool for g in result)
    assert len(set(result)) == step_size  # No duplicates


def test_ea_select_batch_quality_plus_diversity(sample_grower_data):
    """Test _EA_select_batch_quality_plus_diversity selects the right number of diverse samples."""
    grower = ConcreteGrower(**sample_grower_data)

    x_train = sample_grower_data['x_train_val'].iloc[:5]
    x_val = sample_grower_data['x_train_val'].iloc[5:]
    quality = pd.Series(np.random.rand(15), index=x_val.index)

    selected = grower._EA_select_batch_quality_plus_diversity(
        x_train, x_val, quality, step_size=3,
        pop_size=10, n_gen=5, random_state=42,
    )

    assert len(selected) == 3
    assert all(idx in x_val.index for idx in selected)
    assert len(set(selected)) == 3  # No duplicates


# ============================================================
# Vendi Score Tests
# ============================================================

def test_entropy_q_shannon():
    """Test entropy_q computes Shannon entropy (q=1)."""
    w = np.array([0.25, 0.25, 0.25, 0.25])
    result = entropy_q(w, q=1)

    expected = np.log(4)
    assert np.isclose(result, expected)


def test_entropy_q_renyi():
    """Test entropy_q computes Rényi entropy (q!=1)."""
    w = np.array([0.5, 0.3, 0.2])
    result = entropy_q(w, q=2)

    assert result > 0


def test_entropy_q_handles_zeros():
    """Test entropy_q ignores zero weights."""
    w = np.array([0.5, 0.5, 0.0, 0.0])
    result = entropy_q(w, q=1)

    expected = np.log(2)
    assert np.isclose(result, expected)


def test_vendi_scorefast():
    """Test vendi_scorefast computes diversity score."""
    np.random.seed(42)
    X = np.random.randn(10, 5)

    score = vendi_scorefast(X, q=1, normalize=True)

    assert score > 0
    assert score <= 10


def test_vendi_scorefast_identical_samples():
    """Test vendi_scorefast returns near-1 score for identical samples."""
    X = np.ones((5, 3))

    score = vendi_scorefast(X, q=1, normalize=True)

    assert score < 2.0


def test_q_vendi_score():
    """Test q_vendi_score combines quality and diversity."""
    vendi = 5.0
    quality_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    score = q_vendi_score(vendi, quality_scores)

    expected = np.mean(quality_scores) * vendi
    assert np.isclose(score, expected)


def test_q_vendi_scorefast():
    """Test q_vendi_scorefast for single sample."""
    vendi = 3.0
    quality = 0.5

    score = q_vendi_scorefast(vendi, quality)

    assert score == vendi * quality


# ============================================================
# Quality-Diversity Selection Tests
# ============================================================

def test_bestindices_qvendiscorebatch():
    """Test Bestindices_qVendiScoreBatch selects diverse, high-quality samples."""
    np.random.seed(42)

    trainingdata = pd.DataFrame(np.random.randn(10, 5))
    candidates = pd.DataFrame(np.random.randn(20, 5))
    diffs = pd.Series(np.random.rand(20), index=candidates.index)

    selected = Bestindices_qVendiScoreBatch(trainingdata, candidates, diffs, step_size=5)

    assert len(selected) == 5
    assert all(idx in candidates.index for idx in selected)


def test_bestindices_qvendiscorebatch_with_dataframe_diffs():
    """Test Bestindices_qVendiScoreBatch handles multi-column DataFrame diffs."""
    np.random.seed(42)

    trainingdata = pd.DataFrame(np.random.randn(10, 5))
    candidates = pd.DataFrame(np.random.randn(20, 5))
    diffs = pd.DataFrame({
        'target1': np.random.rand(20),
        'target2': np.random.rand(20),
    }, index=candidates.index)

    selected = Bestindices_qVendiScoreBatch(trainingdata, candidates, diffs, step_size=5)

    assert len(selected) == 5


def test_bestindices_qvendiscorefast():
    """Test Bestindices_qVendiScoreFast approximation."""
    np.random.seed(42)

    trainingdata = pd.DataFrame(np.random.randn(10, 5))
    candidates = pd.DataFrame(np.random.randn(15, 5))
    diffs = pd.Series(np.random.rand(15), index=candidates.index)

    selected = Bestindices_qVendiScoreFast(trainingdata, candidates, diffs, step_size=3)

    assert len(selected) == 3
    assert all(idx in candidates.index for idx in selected)


@pytest.mark.slow
def test_bestindices_qvendiscore_greedy():
    """Test Bestindices_qVendiScore greedy selection."""
    np.random.seed(42)

    trainingdata = pd.DataFrame(np.random.randn(5, 3))
    candidates = pd.DataFrame(np.random.randn(8, 3))
    diffs = pd.Series(np.random.rand(8), index=candidates.index)

    selected = Bestindices_qVendiScore(trainingdata, candidates, diffs, step_size=2)

    assert len(selected) == 2
    assert all(idx in candidates.index for idx in selected)


# ============================================================
# Integration Test
# ============================================================

def test_run_al_loop(sample_grower_data):
    """Test _run_al_loop runs to completion, selecting batches and saving results."""
    from sklearn.linear_model import LinearRegression

    grower = ConcreteGrower(
        **sample_grower_data,
        model1=LinearRegression(),
        model2=LinearRegression(),
        flt_step_frac=0.2,
        int_random_seed=42,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        grower.str_save_dir = tmpdir
        result = grower._run_al_loop()

    assert result is True
