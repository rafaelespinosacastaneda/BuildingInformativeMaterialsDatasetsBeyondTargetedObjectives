"""
Active Learning Growth Strategies with Support for ALIGNN Models.

This module provides a unified abstract base class for implementing various active learning
strategies (QBC, Random Sampling, Multi-objective, Diversity-based, EA-based, etc.) with 
optional ALIGNN (Atomistic Line Graph Neural Network) model evaluation.

ALIGNN Model Support:
====================

The ActiveLearningGrower class and its subclasses support optional evaluation using ALIGNN
models alongside traditional ML models (e.g., Random Forest, XGBoost).

Requirements for using ALIGNN models:
1. An ALIGNN model instance (pre-trained or prepared for fitting)
2. sample_ids: A pandas Series mapping training data indices to structure reference IDs
3. dataset: A dataset identifier (e.g., 'mp21', 'jarvis18')
4. A pickle file containing ALIGNN training data: 'alldataRAW_{dataset}_ALIGNN.pkl'

Example Usage:
--------------
    from alignn.models.alignn import ALIGNN
    from multiobj_al.policies.random_policy import RandomPolicy

    # Load or create ALIGNN model
    alignn_model = ALIGNN.from_pretrained(model_name='your_model')

    # Prepare data with flexible targets
    x_train_val = pd.read_csv('features_train_val.csv')
    sample_ids = pd.read_csv('sample_ids.csv', index_col=0)['reference_id']

    # Define targets as dictionaries
    y_targets_train_val = {
        'e_form': pd.Series(...),
        'bulk_modulus': pd.Series(...),
        'bandgap': pd.Series(...)
    }
    y_targets_test = {
        'e_form': pd.Series(...),
        'bulk_modulus': pd.Series(...),
        'bandgap': pd.Series(...)
    }

    # Create active learning grower with ALIGNN
    al_grower = RandomPolicy(
        model1=rf_model,
        model2=xgb_model,
        x_train_val=x_train_val,
        y_targets_train_val=y_targets_train_val,
        x_test=x_test,
        y_targets_test=y_targets_test,
        alignn_model=alignn_model,       # Provide ALIGNN model
        sample_ids=sample_ids,            # Provide sample IDs mapping
        dataset='mp21',                   # Dataset identifier
        str_save_dir='./results',
        int_random_seed=42
    )

    # Run active learning loop
    al_grower.grow()

When alignn_model is provided, the active learning loop will:
- Fit the ALIGNN model on training data (if enough ALIGNN-compatible samples exist)
- Evaluate ALIGNN predictions on validation data
- Store ALIGNN metrics alongside standard model metrics
- Output results to CSV files with both model types' performance
"""

import logging
import numpy as np
import scipy.linalg
import pandas as pd
import os
import time
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any, Union
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize

from ..myfunc import get_scores


# Configure logging for this module
logger = logging.getLogger(__name__)


class ActiveLearningGrower(ABC):
    """
    Unified abstract base class for active-learning grow strategies.
    
    Provides a template method pattern for implementing various active learning strategies
    (e.g., QBC, Random, Multi-objective, Diversity-based) with support for:
    - Multi-property target learning (e.g., formation energy, bulk modulus, bandgap)
    - Dual-model evaluation and disagreement-based selection
    - Optional ALIGNN (Atomistic Line Graph Neural Network) model evaluation
    
    Subclasses must implement:
    - _select_next_batch(): Define the sample selection strategy
    - grow(): Initialize parameters and trigger the active learning loop
    """

    def __init__(
        self,
        x_train_val: pd.DataFrame,
        y_targets_train_val: Dict[str, pd.Series],
        x_test: pd.DataFrame,
        y_targets_test: Dict[str, pd.Series],
        str_save_dir: str = '.',
        str_save_name: str = 'results',
        str_model1_name: str = 'model1',
        str_model2_name: str = 'model2',
        flt_step_frac: Union[float, List[float]] = 0.01,
        n_al_steps: Optional[int] = None,
        int_random_seed: int = 0,
        alignn_model=None,
        sample_ids=None,
        dataset: Optional[str] = None,
        n_round: int = 6,
        n_train_ratio: int = 2,
        property_labels: Optional[Dict[str, str]] = None,
        apply_globals: bool = False,

    ):
        """
        Initialize the Active Learning Grower.

        Parameters
        ----------
        x_train_val : pd.DataFrame
            Training+validation feature data
        y_targets_train_val : Dict[str, pd.Series]
            Dictionary mapping target names to target vectors for training+validation
            Example: {'e_form': Series(...), 'bulk_modulus': Series(...), 'bandgap': Series(...)}
        x_test : pd.DataFrame
            Test feature data
        y_targets_test : Dict[str, pd.Series]
            Dictionary mapping target names to test target vectors
        str_save_dir : str, default='.'
            Directory to save results
        str_save_name : str, default='results'
            Name prefix for result files
        str_model1_name, str_model2_name : str
            Names for the two models (for result file naming)
        flt_step_frac : float or list of float, default=0.01
            Fraction of data to add per AL iteration. A single float is used for
            initialization and every selection. A list defines a schedule: index 0 sets
            only the initial batch size, index 1 is used for the first selection, and so
            on; the last value is repeated for all remaining selections.
        int_random_seed : int, default=0
            Random seed for reproducibility
        alignn_model : object, optional
            ALIGNN (Atomistic Line Graph Neural Network) model instance for evaluation.

            When provided, the active learning loop will also evaluate ALIGNN predictions
            on the training data. Requires:
            - sample_ids: mapping from feature indices to structure references
            - dataset: identifier for the dataset (used to load ALIGNN training data)
            - An 'alldataRAW_{dataset}_ALIGNN.pkl' pickle file containing structure data

            Example:
                from alignn.models.alignn import ALIGNN
                alignn = ALIGNN(...)
                grower = MyPolicy(
                    ...,
                    alignn_model=alignn,
                    sample_ids=id_mapping,
                    dataset='mp21'
                )
        sample_ids : pd.Series, optional
            Mapping from training data index to structure reference IDs.
            Required when alignn_model is not None.

            Format: pd.Series where index is training data row index and values
            are reference IDs (e.g., 'mp-123', 'jarvis-456')
        dataset : str, optional
            Dataset identifier (e.g., 'mp21', 'jarvis18') used to load ALIGNN training data.
            Required when alignn_model is not None.
            The system expects a pickle file: 'alldataRAW_{dataset}_ALIGNN.pkl'
        n_round : int, default=6
            Decimal rounding for metric outputs
        n_train_ratio : int, default=2
            Decimal rounding for train/total ratio outputs
        property_labels : Dict[str, str], optional
            Display labels for properties in logging.
            If not provided, uses uppercase of target names.
        apply_globals : bool, default=False
            If True, set module-level globals (legacy compatibility mode)
        """
        # Validate inputs
        if not y_targets_train_val:
            raise ValueError("y_targets_train_val must contain at least one target")

        if set(y_targets_train_val.keys()) != set(y_targets_test.keys()):
            raise ValueError("Train and test targets must have the same keys")

        self.n_round = n_round
        self.n_train_ratio = n_train_ratio

        # Extract target names (properties) from the dictionary
        self.properties = list(y_targets_train_val.keys())

        # Create property labels (uppercase of target names if not provided)
        if property_labels is None:
            self.property_labels = {prop: prop.upper().replace('_', ' ') for prop in self.properties}
        else:
            self.property_labels = property_labels

        # Create target_map for compatibility (maps full name to short name)
        self.target_map = {prop: prop for prop in self.properties}

        # Optional compatibility mode for legacy module-level helpers.
        if apply_globals:
            global N_ROUND, N_TRAIN_RATIO, PROPERTIES, PROPERTY_LABELS
            N_ROUND = self.n_round
            N_TRAIN_RATIO = self.n_train_ratio
            PROPERTIES = self.properties
            PROPERTY_LABELS = self.property_labels

        self.x_train_val = x_train_val
        self.y_targets_train_val = y_targets_train_val
        self.x_test = x_test
        self.y_targets_test = y_targets_test
        self.str_save_dir = str_save_dir
        self.str_save_name = str_save_name
        self.str_model1_name = str_model1_name
        self.str_model2_name = str_model2_name
        # Normalize to a list so the grow loop can apply a per-iteration schedule.
        self.step_fracs: List[float] = (
            list(flt_step_frac) if isinstance(flt_step_frac, (list, tuple))
            else [float(flt_step_frac)]
        )
        # Keep scalar alias so _initialize_al_loop (which uses the first fraction) is unchanged.
        self.flt_step_frac = self.step_fracs[0]
        # When set, the loop stops after exactly this many batch selections.
        # None means run until the pool is exhausted.
        self.n_al_steps = n_al_steps
        self.int_random_seed = int_random_seed
        self.alignn_model = alignn_model
        self.sample_ids = sample_ids
        self.dataset = dataset


    def _prepare_y_trainval(self) -> pd.DataFrame:
        """Combine all target training/validation series into a single DataFrame."""
        return pd.DataFrame({
            f'{target}TrainVal': series
            for target, series in self.y_targets_train_val.items()
        })

    def _initialize_al_loop(self):
        """
        Initialize active learning loop parameters.

        Sets random seeds, calculates initial training size and step size based on
        flt_step_frac, and randomly selects initial training indices.

        Returns
        -------
        Tuple[int, int, List[int]]
            (initial_train_size, step_size, initial_indices)
        """
        np.random.seed(self.int_random_seed)
        random.seed(self.int_random_seed)
        initial_train_size = np.rint(len(self.x_train_val) * self.flt_step_frac).astype(int)
        step_size = np.rint(len(self.x_train_val) * self.flt_step_frac).astype(int)
        initial_indices = np.random.choice(self.x_train_val.index, initial_train_size, replace=False).tolist()
        return initial_train_size, step_size, initial_indices

    def _log_iteration_info(self, train_size: int, total_size: int, time_elapsed: Optional[float] = None):
        logger.info(f'Training set size: {train_size}')
        logger.info(f'Training set percentage: {train_size/total_size:.1%}\\n')
        if time_elapsed is not None:
            logger.info(f'Time to run loop: {round(time_elapsed, 2)} seconds')

    def _min_max_standardize(self, array_to_standardize, eps=1e-12):
        """
        Min-max normalize an array to [0, 1] range.

        Parameters
        ----------
        array_to_standardize : array-like
            Array to normalize
        eps : float, default=1e-12
            Small constant to avoid division by zero

        Returns
        -------
        np.ndarray
            Normalized array, or zeros if range is too small
        """
        arr = np.asarray(array_to_standardize, dtype=np.float64)
        min_ = float(np.min(arr))
        max_ = float(np.max(arr))
        if max_ - min_ < eps:
            return np.zeros_like(arr)
        standarized_array = (arr - min_) / (max_ - min_)
        return standarized_array


    def _calculate_multi_property_scores(
        self,
        model,
        xTrain,
        y_train_dict,
        xTest,
        y_test_dict,
        xVal=None,
        y_val_dict=None,
        model_name='Model',
        log_results=True,
        alignn_model=None,
        sample_ids=None,
        dataset=None,
    ):
        """
        Calculate scores for all properties using the model.

        Parameters
        ----------
        model : object
            ML model with fit/predict interface
        xTrain : pd.DataFrame
            Training features
        y_train_dict : dict
            Dictionary with {target}Train keys mapping to training targets
        xTest : pd.DataFrame
            Test features
        y_test_dict : dict
            Dictionary with target names mapping to test targets
        xVal : pd.DataFrame, optional
            Validation features
        model_name : str
            Name for logging
        log_results : bool
            Whether to log results
        alignn_model : object, optional
            ALIGNN model instance
        sample_ids : pd.Series, optional
            Sample ID mapping
        dataset : str, optional
            Dataset identifier

        Returns
        -------
        dict
            Results dictionary with keys being property names
        """
        results: Dict[str, Dict[str, Any]] = {}

        for prop_name in self.properties:
            train_key = f'{prop_name}Train'
            val_key = f'{prop_name}Val'

            if train_key not in y_train_dict or prop_name not in y_test_dict:
                logger.warning(f"Skipping property {prop_name}: missing in train or test dict")
                continue

            if log_results:
                label = self.property_labels.get(prop_name, prop_name.upper())
                logger.info(f"{model_name} METRICS FOR {label}")

            results[prop_name] = get_scores(
                model, xTrain, y_train_dict[train_key],
                xTest, y_test_dict[prop_name], X_val=xVal,
                y_val=y_val_dict[val_key] if y_val_dict is not None and val_key in y_val_dict else None,
                alignn_model=alignn_model, sample_ids=sample_ids,
                dataset=dataset, train_target=prop_name,
            )
        return results

    def _create_results_dataframe(self, train_size: int, total_size: int, model_results: Dict[str, Dict[str, Any]]):
        data = {
            'train_ratio': [round(train_size / total_size, self.n_train_ratio)],
            'train_size': [train_size],
        }
        for prop in self.properties:
            if prop not in model_results:
                continue
            metrics = model_results[prop]
            data[f'mae_{prop}'] = [round(metrics['model_maes'], self.n_round)]
            data[f'rmse_{prop}'] = [round(metrics['model_rmse'], self.n_round)]
            data[f'r2_{prop}'] = [round(metrics['model_r2'], self.n_round)]
            if 'alignn_maes' in metrics:
                alignn_mae = metrics['alignn_maes']
                alignn_rmse = metrics['alignn_rmse']
                alignn_r2 = metrics['alignn_r2']
                data[f'alignn_mae_{prop}'] = [round(alignn_mae, self.n_round) if not (isinstance(alignn_mae, float) and np.isnan(alignn_mae)) else np.nan]
                data[f'alignn_rmse_{prop}'] = [round(alignn_rmse, self.n_round) if not (isinstance(alignn_rmse, float) and np.isnan(alignn_rmse)) else np.nan]
                data[f'alignn_r2_{prop}'] = [round(alignn_r2, self.n_round) if not (isinstance(alignn_r2, float) and np.isnan(alignn_r2)) else np.nan]
        return pd.DataFrame(data)

    def _save_results(
        self,
        df_model1,
        df_model2,
        train_indices,
        save_dir='.',
        save_name='results',
        model1_name='model_1',
        model2_name='model_2',
    ):
        if not os.path.exists(save_dir):
            logger.info(f'Creating directory: {save_dir}')
            os.makedirs(save_dir)

        save_path_1 = os.path.join(save_dir, f'{save_name}_{model1_name}.csv')
        save_path_2 = os.path.join(save_dir, f'{save_name}_{model2_name}.csv')
        idx_save_path = os.path.join(save_dir, f'{save_name}_trainIndices.csv')

        df_model1.to_csv(save_path_1)
        df_model2.to_csv(save_path_2)
        pd.DataFrame(train_indices, columns=['trainIndices']).to_csv(idx_save_path)
        return save_path_1, save_path_2, idx_save_path

    def _combine_normalized_differences(self, results_model1, results_model2, target1: str, target2: str):
        prop1 = self.target_map[target1]
        prop2 = self.target_map[target2]
        diff_1 = abs(results_model1[prop1]['model_pred'] - results_model2[prop1]['model_pred'])
        diff_2 = abs(results_model1[prop2]['model_pred'] - results_model2[prop2]['model_pred'])
        return self._min_max_standardize(diff_1) + self._min_max_standardize(diff_2)

    def _combine_all_properties_differences(self, results_model1, results_model2):
        combined = None
        for prop in self.properties:
            diff = abs(results_model1[prop]['model_pred'] - results_model2[prop]['model_pred'])
            norm = self._min_max_standardize(diff)
            combined = norm if combined is None else combined + norm
        return combined

    def _normalize_save_dir(self, save_dir: str) -> str:
        if save_dir != '' and save_dir[-1] != '/':
            return save_dir + '/'
        return save_dir

    def _split_train_val(self, x_train_val, y_train_val, train_idx):
        """
        Split combined train+validation data into separate train and validation sets.

        Parameters
        ----------
        x_train_val : pd.DataFrame
            Combined feature data
        y_train_val : pd.DataFrame
            Combined target data with columns {target}TrainVal for each target
        train_idx : List[int]
            Indices to use for training set

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (x_train, x_val, y_train, y_val) where y_train and y_val are DataFrames
            with columns {target}Train and {target}Val respectively
        """
        # Extract each target's train/val series
        target_names = [col.replace('TrainVal', '') for col in y_train_val.columns if col.endswith('TrainVal')]

        x_train = x_train_val.loc[train_idx]
        x_val = x_train_val.drop(train_idx)

        # Build y_train and y_val DataFrames dynamically
        y_train_dict = {}
        y_val_dict = {}

        for target in target_names:
            target_series = y_train_val[f'{target}TrainVal']
            y_train_dict[f'{target}Train'] = target_series.loc[train_idx]
            y_val_dict[f'{target}Val'] = target_series.drop(train_idx)

        y_train = pd.DataFrame(y_train_dict)
        y_val = pd.DataFrame(y_val_dict)

        return x_train, x_val, y_train, y_val

    # ============================================================
    # NSGA-II Multi-Objective Optimization Helpers
    # ============================================================
    # These methods implement the NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    # for multi-objective optimization. Used by EA-based active learning subclasses
    # to balance quality and diversity objectives when selecting training samples.
    # ============================================================

    def _fast_non_dominated_sort(self, F):
        """
        NSGA-II fast non-dominated sorting algorithm.

        Sorts population into Pareto fronts based on dominance relationships.

        Parameters
        ----------
        F : np.ndarray, shape (pop, M)
            Fitness matrix where higher values are better. Each row is an individual,
            each column is an objective.

        Returns
        -------
        Tuple[List[List[int]], np.ndarray]
            (fronts, rank) where fronts is a list of lists containing indices for each
            Pareto front, and rank is an array of front numbers for each individual
        """
        pop = F.shape[0]
        S = [[] for _ in range(pop)]
        n = np.zeros(pop, dtype=int)
        rank = np.zeros(pop, dtype=int)
        fronts = [[]]

        for p in range(pop):
            for q_idx in range(pop):
                if p == q_idx:
                    continue
                if np.all(F[p] >= F[q_idx]) and np.any(F[p] > F[q_idx]):
                    S[p].append(q_idx)
                elif np.all(F[q_idx] >= F[p]) and np.any(F[q_idx] > F[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q_idx in S[p]:
                    n[q_idx] -= 1
                    if n[q_idx] == 0:
                        rank[q_idx] = i + 1
                        next_front.append(q_idx)
            i += 1
            fronts.append(next_front)

        fronts.pop()
        return fronts, rank

    def _crowding_distance(self, F, front):
        """
        Calculate crowding distance for diversity preservation in NSGA-II.

        Measures how close an individual is to its neighbors in objective space.
        Higher values indicate more isolated (diverse) solutions.

        Parameters
        ----------
        F : np.ndarray, shape (pop, M)
            Fitness matrix
        front : List[int]
            Indices of individuals in the current Pareto front

        Returns
        -------
        Dict[int, float]
            Mapping from individual index to crowding distance (inf for boundary points)
        """
        if len(front) == 0:
            return {}
        if len(front) <= 2:
            return {i: np.inf for i in front}

        crowd = {i: 0.0 for i in front}
        M = F.shape[1]

        for m in range(M):
            vals = np.array([F[i, m] for i in front], dtype=float)
            order = np.argsort(vals)
            sorted_idx = [front[j] for j in order]
            crowd[sorted_idx[0]] = np.inf
            crowd[sorted_idx[-1]] = np.inf
            vmin, vmax = vals[order[0]], vals[order[-1]]
            denom = (vmax - vmin) if (vmax > vmin) else 1.0
            for k in range(1, len(front) - 1):
                crowd[sorted_idx[k]] += (F[sorted_idx[k + 1], m] - F[sorted_idx[k - 1], m]) / denom

        return crowd

    def _tournament_select(self, pop_indices, rank, crowd, rng, k=2):
        """
        Binary tournament selection for NSGA-II.

        Selects the better individual based on dominance rank (lower is better),
        with ties broken by crowding distance (higher is better).

        Parameters
        ----------
        pop_indices : array-like
            Indices of individuals in the population
        rank : array-like
            Pareto front rank for each individual
        crowd : Dict[int, float]
            Crowding distance for each individual
        rng : np.random.Generator
            Random number generator
        k : int, default=2
            Tournament size

        Returns
        -------
        int
            Index of the selected individual
        """
        best = None
        for _ in range(k):
            i = int(rng.choice(pop_indices))
            if best is None:
                best = i
            elif rank[i] < rank[best]:
                best = i
            elif rank[i] == rank[best] and crowd.get(i, 0.0) > crowd.get(best, 0.0):
                best = i
        return best

    def _make_unique(self, chrom, pool_size, rng):
        """
        Repair chromosome by ensuring all genes are unique.

        Replaces duplicate genes with random unused values from the pool.

        Parameters
        ----------
        chrom : array-like
            Chromosome (array of indices)
        pool_size : int
            Size of the available gene pool
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        np.ndarray
            Repaired chromosome with unique genes
        """
        chrom = np.array(chrom, dtype=int)
        used = set()
        for t in range(len(chrom)):
            if int(chrom[t]) in used:
                while True:
                    r = int(rng.integers(0, pool_size))
                    if r not in used:
                        chrom[t] = r
                        break
            used.add(int(chrom[t]))
        return chrom

    def _ordered_crossover(self, a, b, rng):
        """
        Order crossover (OX) operator for permutation-based chromosomes.

        Preserves relative order of genes while mixing genetic material from two parents.

        Parameters
        ----------
        a, b : array-like
            Parent chromosomes
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two offspring chromosomes
        """
        a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
        B = a.size
        if B < 2:
            return a.copy(), b.copy()

        i, j = sorted(rng.choice(np.arange(B), size=2, replace=False))
        if i == j:
            j = min(B, i + 1)

        child1 = np.full(B, -1, dtype=int)
        child2 = np.full(B, -1, dtype=int)
        child1[i:j] = a[i:j]
        child2[i:j] = b[i:j]

        def fill(child, parent):
            taken = set(child[child != -1].tolist())
            remaining = [g for g in parent.tolist() if g not in taken]
            empty_pos = np.where(child == -1)[0]
            m = min(len(empty_pos), len(remaining))
            child[empty_pos[:m]] = remaining[:m]
            if np.any(child == -1):
                left = np.where(child == -1)[0]
                child[left] = rng.choice(parent, size=left.size, replace=True)
            return child

        return fill(child1, b), fill(child2, a)

    def _mutate_swap(self, chrom, pool_size, rng, p_mut=0.2):
        """
        Mutation operator that randomly replaces one gene.

        With probability p_mut, replaces a randomly selected gene with a new random value.

        Parameters
        ----------
        chrom : array-like
            Chromosome to mutate
        pool_size : int
            Size of the gene pool
        rng : np.random.Generator
            Random number generator
        p_mut : float, default=0.2
            Mutation probability

        Returns
        -------
        np.ndarray
            Mutated chromosome
        """
        chrom = chrom.copy()
        if rng.random() < p_mut:
            t = int(rng.integers(0, len(chrom)))
            chrom[t] = int(rng.integers(0, pool_size))
        return chrom

    def _nsga2_evolve(self, fitness_func, n_pool, step_size, pop_size, n_gen, p_cx, p_mut, random_state):
        """
        Run NSGA-II multi-objective genetic algorithm.

        Evolves a population of chromosomes (sample sets) to optimize multiple objectives.
        Returns the best chromosome from the final Pareto front.

        Parameters
        ----------
        fitness_func : callable
            Function that takes a chromosome and returns a tuple of fitness values
        n_pool : int
            Size of the candidate pool
        step_size : int
            Number of samples to select (chromosome size)
        pop_size : int
            Population size for the genetic algorithm
        n_gen : int
            Number of generations to evolve
        p_cx : float
            Crossover probability
        p_mut : float
            Mutation probability
        random_state : int
            Random seed

        Returns
        -------
        np.ndarray or None
            Best chromosome (array of indices) from the final Pareto front,
            or None if step_size is invalid
        """
        rng = np.random.default_rng(random_state)
        B = int(min(step_size, n_pool))
        if B <= 0:
            return None

        pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

        for _ in range(n_gen):
            F = np.array([fitness_func(ind) for ind in pop], dtype=float)
            fronts, rank = self._fast_non_dominated_sort(F)
            crowd = {}
            for fr in fronts:
                crowd.update(self._crowding_distance(F, fr))

            offspring = []
            pop_indices = np.arange(len(pop))
            while len(offspring) < pop_size:
                p1 = pop[self._tournament_select(pop_indices, rank, crowd, rng)]
                p2 = pop[self._tournament_select(pop_indices, rank, crowd, rng)]
                c1, c2 = p1.copy(), p2.copy()
                if rng.random() < p_cx:
                    c1, c2 = self._ordered_crossover(c1, c2, rng)
                c1 = self._make_unique(self._mutate_swap(self._make_unique(c1, n_pool, rng), n_pool, rng, p_mut), n_pool, rng)
                c2 = self._make_unique(self._mutate_swap(self._make_unique(c2, n_pool, rng), n_pool, rng, p_mut), n_pool, rng)
                offspring.append(c1)
                if len(offspring) < pop_size:
                    offspring.append(c2)

            combined = pop + offspring
            F2 = np.array([fitness_func(ind) for ind in combined], dtype=float)
            fronts2, _ = self._fast_non_dominated_sort(F2)
            new_pop = []
            for fr in fronts2:
                if len(new_pop) + len(fr) <= pop_size:
                    new_pop.extend([combined[i] for i in fr])
                else:
                    cd = self._crowding_distance(F2, fr)
                    fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                    new_pop.extend([combined[i] for i in fr_sorted[: pop_size - len(new_pop)]])
                    break
            pop = new_pop

        F_final = np.array([fitness_func(ind) for ind in pop], dtype=float)
        fronts, _ = self._fast_non_dominated_sort(F_final)
        front0 = fronts[0]
        f0 = F_final[front0]
        fmin, fmax = f0.min(axis=0), f0.max(axis=0)
        denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
        score = ((f0 - fmin) / denom).mean(axis=1)
        return pop[front0[int(np.argmax(score))]]

    def _EA_select_batch_quality_plus_diversity(
        self,
        trainingdata: pd.DataFrame,
        candidates: pd.DataFrame,
        q,
        step_size: int,
        do_scale: bool = True,
        normalize_feats: bool = True,
        pop_size: int = 120,
        n_gen: int = 60,
        p_cx: float = 0.9,
        p_mut: float = 0.25,
        random_state: int = 0,
        w_train: float = 0.5,
        w_within: float = 0.5,
    ):
        """
        Select batch using NSGA-II with 2 objectives: quality and diversity.

        Uses multi-objective evolutionary algorithm to find a Pareto-optimal batch
        that balances sample quality (uncertainty/disagreement) with diversity
        (both from training set and within the selected batch).

        Parameters
        ----------
        trainingdata : pd.DataFrame
            Current training feature data
        candidates : pd.DataFrame
            Candidate pool feature data
        q : pd.Series or array-like
            Quality scores for each candidate (higher is better)
        step_size : int
            Number of samples to select
        do_scale : bool, default=True
            Whether to standardize features
        normalize_feats : bool, default=True
            Whether to L2-normalize feature vectors
        pop_size : int, default=120
            NSGA-II population size
        n_gen : int, default=60
            Number of evolutionary generations
        p_cx : float, default=0.9
            Crossover probability
        p_mut : float, default=0.25
            Mutation probability
        random_state : int, default=0
            Random seed
        w_train : float, default=0.5
            Weight for diversity from training set
        w_within : float, default=0.5
            Weight for diversity within selected batch

        Returns
        -------
        List[int]
            Indices of selected samples from candidates
        """
        X_train = trainingdata.values.astype(np.float32, copy=False)
        X_pool = candidates.values.astype(np.float32, copy=False)

        if do_scale:
            scaler = StandardScaler()
            scaler.fit(np.vstack([X_train, X_pool]))
            X_train = scaler.transform(X_train).astype(np.float32, copy=False)
            X_pool = scaler.transform(X_pool).astype(np.float32, copy=False)
        if normalize_feats:
            X_train = normalize(X_train, axis=1)
            X_pool = normalize(X_pool, axis=1)

        n_pool = X_pool.shape[0]

        if isinstance(q, pd.Series):
            qv = q.reindex(candidates.index).values.astype(np.float32)
        else:
            qv = np.asarray(q, dtype=np.float32).reshape(-1)
        qv = np.nan_to_num(qv, nan=0.0, posinf=0.0, neginf=0.0)

        sim_pt = (X_pool @ X_train.T).mean(axis=1).astype(np.float32) if X_train.shape[0] > 0 else np.zeros(n_pool, dtype=np.float32)
        G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)

        def fitness(chrom):
            idx = np.asarray(chrom, dtype=int)
            qual = float(np.mean(qv[idx]))
            div_train = float(np.mean(1.0 - sim_pt[idx]))
            if len(idx) <= 1:
                div_within = 1.0
            else:
                S = G_pp[np.ix_(idx, idx)]
                div_within = float(1.0 - np.mean(S[~np.eye(len(idx), dtype=bool)]))
            div = float(w_train * div_train + w_within * div_within)
            return qual, div

        best = self._nsga2_evolve(fitness, n_pool, step_size, pop_size, n_gen, p_cx, p_mut, random_state)
        return candidates.index[np.asarray(best, dtype=int)].tolist() if best is not None else []

    @abstractmethod
    def _select_next_batch(self, x_train: pd.DataFrame, x_val: pd.DataFrame,
                          results_model1: Dict, results_model2: Dict, 
                          step_size: int) -> List[int]:
        """
        Select next batch of samples to add to training set.
        
        This abstract method must be implemented by each subclass to define
        the sample selection strategy (QBC, random, EA, etc).
        
        Parameters
        ----------
        x_train : pd.DataFrame
            Current training features
        x_val : pd.DataFrame
            Current validation features
        results_model1, results_model2 : Dict
            Dictionary of model predictions/scores for each property
        step_size : int
            Number of samples to select
            
        Returns
        -------
        List[int]
            List of selected sample indices from x_val
        """
        pass

    def _run_al_loop(self) -> bool:
        """
        Unified active learning loop for all strategies.
        
        This template method handles the common AL loop structure:
        - Initialize training set
        - Iteratively:
          - Split train/val
          - Evaluate both models
          - Record results
          - Select next batch (delegated to subclass)
          - Update training set
        - Save final results
        
        Subclasses just need to implement _select_next_batch() to define
        their selection strategy.
        
        Returns
        -------
        bool
            True on successful completion
        """
        # Initialize results tracking
        model1_rows: list = []
        model2_rows: list = []
        
        # Initialize AL loop
        temp_train_size, step_size, train_indices = self._initialize_al_loop()

        # Prepare target data
        y_train_val = self._prepare_y_trainval()
        y_test = self.y_targets_test  # Use flexible targets dictionary
        total_samples = len(self.x_train_val)

        # Start at 1 so fracs[0] is used only for initialization;
        # the first selection uses fracs[1] (or fracs[0] when list has one element).
        n_iter = 1

        # Main active learning loop
        while True:
            time_start = time.time()

            # Split into train/val
            x_train, x_val, y_train_dict, y_val_dict = self._split_train_val(
                self.x_train_val, y_train_val, train_indices
            )

            # Log current iteration info
            self._log_iteration_info(len(x_train), total_samples)

            # Evaluate both models
            results_model1 = self._calculate_multi_property_scores(
                self.model1, x_train, y_train_dict, self.x_test, y_test, x_val,
                y_val_dict=y_val_dict,
                model_name='MODEL1',
                alignn_model=self.alignn_model,
                sample_ids=self.sample_ids,
                dataset=getattr(self, 'dataset', None),
            )
            results_model2 = self._calculate_multi_property_scores(
                self.model2, x_train, y_train_dict, self.x_test, y_test, x_val,
                y_val_dict=y_val_dict,
                model_name='MODEL2',
                alignn_model=None,
                sample_ids=self.sample_ids,
                dataset=getattr(self, 'dataset', None),
            )

            # Store results for this iteration
            model1_rows.append(self._create_results_dataframe(len(x_train), total_samples, results_model1))
            model2_rows.append(self._create_results_dataframe(len(x_train), total_samples, results_model2))

            # Stop after evaluating the full training set.
            if temp_train_size >= total_samples:
                break

            # Stop after the requested number of batch selections (milestone mode).
            if self.n_al_steps is not None and n_iter > self.n_al_steps:
                break

            # Recompute step size from the schedule for this iteration.
            current_frac = self.step_fracs[min(n_iter, len(self.step_fracs) - 1)]
            step_size = max(1, np.rint(total_samples * current_frac).astype(int))
            n_iter += 1

            # Select next batch (delegated to subclass)
            selected_indices = self._select_next_batch(
                x_train, x_val, results_model1, results_model2, step_size
            )

            if not selected_indices:
                logger.info('No points selected. Stopping.')
                break

            # Update training set
            train_indices.extend(selected_indices)
            temp_train_size = len(train_indices)

            # Log timing information
            self._log_iteration_info(len(x_train), total_samples, time.time() - time_start)
        
        # Save results
        df_model1_results = pd.concat(model1_rows, ignore_index=True)
        df_model2_results = pd.concat(model2_rows, ignore_index=True)
        self._save_results(
            df_model1_results, df_model2_results, train_indices,
            save_dir=self.str_save_dir,
            save_name=self.str_save_name,
            model1_name=self.str_model1_name,
            model2_name=self.str_model2_name,
        )
        
        return True

# ============================================================
# Abstract Methods for Subclass Implementation
# ============================================================
# Subclasses must implement these methods to define their specific
# active learning strategy (QBC, random, diversity-based, EA-based, etc.)
# ============================================================

    @abstractmethod
    def grow(self) -> bool:
        """
        Run active learning.
        
        Each subclass should implement this to initialize parameters
        and call self._run_al_loop() to execute the common AL loop.
        
        Returns
        -------
        bool
            True on successful completion
        """
        pass



# ============================================================
# Vendi Score Functions for Diversity Measurement
# ============================================================
# These functions compute Vendi scores (based on Rényi entropy of eigenvalues)
# to measure diversity in feature space. Used for diversity-based active learning
# sample selection strategies.
# ============================================================
def entropy_q(w, q=1):
    """
    Compute generalized Rényi entropy.

    Parameters
    ----------
    w : array-like
        Probability distribution or eigenvalue weights
    q : float, default=1
        Entropy order. q=1 gives Shannon entropy, q!=1 gives Rényi entropy.

    Returns
    -------
    float
        Entropy value
    """
    w = w[w > 0]
    w = w / np.sum(w)
    if q == 1:
        return -np.sum(w * np.log(w))
    else:
        return (1 / (1 - q)) * np.log(np.sum(w**q))


def vendi_scorefast(X, q=1, normalize=True):
    """
        Compute Vendi score (diversity measure) for a dataset.

        Vendi score quantifies diversity by computing the effective number of distinct
        elements based on the eigenvalues of the similarity matrix. Higher scores indicate
        more diverse datasets.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        q : float, default=1
            Entropy order for Rényi entropy
        normalize : bool, default=True
            Whether to L2-normalize feature vectors before computing similarity

        Returns
        -------
        float
            Vendi score (effective number of distinct samples)
    """

    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    S = X @ X.T
    w = scipy.linalg.eigvalsh(S / n)
    w = w[w > 0]
    return np.exp(entropy_q(w, q=q))


def q_vendi_score(vendi_score, quality_scores):
    """
    Combine quality and diversity into a single score.

    Parameters
    ----------
    vendi_score : float
        Diversity measure from vendi_scorefast
    quality_scores : array-like
        Quality scores for samples (e.g., uncertainty or disagreement)

    Returns
    -------
    float
        Combined quality-diversity score (mean_quality × vendi_score)
    """
    quality_mean = np.mean(quality_scores)
    return quality_mean * vendi_score


def q_vendi_scorefast(vendi_score, quality):
    """
    Fast quality-diversity score for single samples.

    Parameters
    ----------
    vendi_score : float
        Diversity measure
    quality : float
        Quality score for a single sample

    Returns
    -------
    float
        Combined score (quality × vendi_score)
    """
    return quality * vendi_score

# ============================================================
# Quality-Diversity Sample Selection Functions
# ============================================================
# These functions implement greedy and batch selection strategies that balance
# quality (uncertainty/error) and diversity (Vendi score) for active learning.
# - Bestindices_qVendiScore: Greedy sequential selection (exact but slow)
# - Bestindices_qVendiScoreFast: Fast batch ranking approximation
# - Bestindices_qVendiScoreBatch: Vectorized cosine similarity approximation
# ============================================================
def Bestindices_qVendiScore(trainingdata, candidates, diffs, step_size, normalize=True):
    """
    Greedy sequential selection maximizing q-Vendi score (EXACT but SLOW).

    Iteratively selects samples one at a time, each time choosing the candidate
    that maximizes the combined quality-diversity (q-Vendi) score when added to
    the current training set.

    Parameters
    ----------
    trainingdata : pd.DataFrame
        Current training feature data
    candidates : pd.DataFrame
        Candidate pool feature data
    diffs : pd.Series
        Quality scores (uncertainty/disagreement) for each candidate
    step_size : int
        Number of samples to select
    normalize : bool, default=True
        Whether to L2-normalize features for Vendi score computation

    Returns
    -------
    List[int]
        Indices of selected samples
    """
    selected_indices = []
    quality_scores = []

    for step in range(step_size):
        best_score = -np.inf
        best_index = None

        for idx, row in candidates.iterrows():
            # Compute temporary quality
            q_val = diffs.loc[idx] # or adapt depending on diffs shape

            # Temporarily add this candidate to current training data
            temp_samples = pd.concat([trainingdata, row.to_frame().T])
            vendi = vendi_scorefast(temp_samples.values, normalize=normalize)

            # Temporary quality list (including this one)
            temp_quals = quality_scores + [q_val]
            qv = q_vendi_score(vendi, temp_quals)

            if qv > best_score:
                best_score = qv
                best_index = idx

        # Add best candidate
        logger.info('Candidate added')
        selected_indices.append(best_index)
        quality_scores.append(diffs.loc[best_index])
        trainingdata = pd.concat([trainingdata, candidates.loc[[best_index]]])
        candidates = candidates.drop(best_index)

    return selected_indices


#This is an approximation of the greedy approach
def Bestindices_qVendiScoreFast(trainingdata, candidates, diffs, step_size, normalize=True):
    """
    Fast batch ranking selection using q-Vendi score approximation.

    Computes q-Vendi scores for all candidates in parallel (assuming independent
    additions) and selects the top step_size candidates. Faster than greedy but
    less accurate since it doesn't account for interactions between selected samples.

    Parameters
    ----------
    trainingdata : pd.DataFrame
        Current training feature data
    candidates : pd.DataFrame
        Candidate pool feature data
    diffs : pd.Series or pd.DataFrame
        Quality scores for each candidate (averaged if DataFrame)
    step_size : int
        Number of samples to select
    normalize : bool, default=True
        Whether to L2-normalize features

    Returns
    -------
    List[int]
        Indices of selected samples
    """
    # ------------------------------------------------------------
    # 2. Prepare quality scores
    # ------------------------------------------------------------
    if isinstance(diffs, pd.Series):
        quality_series = diffs
    else:
        quality_series = diffs.mean(axis=1)  # mean across targets if multiple columns

    # ------------------------------------------------------------
    # 3. Compute ΔVendi and q-Vendi scores for each candidate
    # ------------------------------------------------------------
    qv_scores = []
    for idx, row in candidates.iterrows():
        # Compute vendi for training + this candidate
        temp_samples = pd.concat([trainingdata, row.to_frame().T])
        vendi_new = vendi_scorefast(temp_samples.values, normalize=normalize)

        # Compute combined q-Vendi score
        q_val = quality_series.loc[idx]
        qv = q_vendi_score(vendi_new,  q_val)

        qv_scores.append(qv)

    # ------------------------------------------------------------
    # 4. Rank candidates by q-Vendi score and select top step_size
    # ------------------------------------------------------------
    results = pd.DataFrame({
        "idx": candidates.index,
        "qv": qv_scores
    }).set_index("idx")

    best_indices = results.nlargest(step_size, "qv").index.tolist()

    logger.info(f"✅ Selected {len(best_indices)} samples with highest ΔVendi × Quality scores.")
    return best_indices



def Bestindices_qVendiScoreBatch(trainingdata, candidates, diffs, step_size, do_normalize=True):
    """
    Vectorized batch selection using cosine similarity approximation (FASTEST).

    Approximates diversity gain (ΔVendi) using mean cosine similarity to training set
    rather than computing full Vendi scores. Much faster than exact methods while
    maintaining reasonable quality.

    Diversity approximation: ΔVendi ≈ 1 - mean_cosine_similarity(candidate, training_set)
    Final score: diversity × quality

    Parameters
    ----------
    trainingdata : pd.DataFrame
        Current training feature data
    candidates : pd.DataFrame
        Candidate pool feature data
    diffs : pd.Series or pd.DataFrame
        Quality scores for each candidate (averaged if DataFrame)
    step_size : int
        Number of samples to select
    do_normalize : bool, default=True
        Whether to L2-normalize features before computing similarity

    Returns
    -------
    List[int]
        Indices of selected samples
    """


    # ------------------------------------------------------------
    # 1. Normalize embeddings
    # ------------------------------------------------------------
    X_train = trainingdata.values
    X_cand = candidates.values

    if do_normalize:
        X_train = normalize(X_train, axis=1)
        X_cand = normalize(X_cand, axis=1)

    # ------------------------------------------------------------
    # 2. Compute cosine similarities (batched)
    # ------------------------------------------------------------
    # Shape: (n_candidates, n_train)
    sims = X_cand @ X_train.T
    mean_sim = sims.mean(axis=1)  # average similarity to training set

    # ------------------------------------------------------------
    # 3. Approximate ΔVendi (diversity gain)
    # ------------------------------------------------------------
    delta_vendi = 1.0 - mean_sim  # higher = more diverse

    # ------------------------------------------------------------
    # 4. Get quality (uncertainty/error) for each candidate
    # ------------------------------------------------------------
    if isinstance(diffs, pd.Series):
        quality_vals = diffs.values
    else:
        quality_vals = diffs.mean(axis=1).values

    # ------------------------------------------------------------
    # 5. Combine diversity and quality
    # ------------------------------------------------------------
    qv_scores =  delta_vendi * quality_vals  # same spirit as q_vendi_score

    # ------------------------------------------------------------
    # 6. Select top step_size candidates
    # ------------------------------------------------------------
    results = pd.DataFrame({
        "idx": candidates.index,
        "qv": qv_scores
    }).set_index("idx")

    best_indices = results.nlargest(step_size, "qv").index.tolist()

    logger.info(f"✅ Selected {len(best_indices)} candidates with highest approximate ΔVendi × Quality scores.")
    return best_indices

