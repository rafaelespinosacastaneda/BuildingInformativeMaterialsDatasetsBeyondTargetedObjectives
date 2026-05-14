from .active_learning_grower import ActiveLearningGrower
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, normalize

logger = logging.getLogger(__name__)


class DiversityPolicy(ActiveLearningGrower):
    """
    Evolutionary Algorithm (EA) based active learning strategy.

    Supports three modes:
    - Two-target QBC without diversity (NSGA-II, 2 objectives)
    - Two-target QBC with diversity (NSGA-II, 3 objectives)
    - Single-target QBC with diversity (single-objective EA)
    """

    def __init__(self, model1, model2, **kwargs):
        self.target1 = kwargs.pop('target1', None)
        self.target2 = kwargs.pop('target2', None)
        self.target = kwargs.pop('target', None)
        self.use_diversity = kwargs.pop('use_diversity', True)
        self.pop_size = kwargs.pop('pop_size', 120)
        self.n_gen = kwargs.pop('n_gen', 60)
        self.p_cx = kwargs.pop('p_cx', 0.9)
        self.p_mut = kwargs.pop('p_mut', 0.25)
        self.w_train = kwargs.pop('w_train', 0.5)
        self.w_within = kwargs.pop('w_within', 0.5)
        self.lambda_div = kwargs.pop('lambda_div', 0.5)
        # Remove deprecated parameters that are now derived from y_targets
        kwargs.pop('outcome_names', None)
        kwargs.pop('properties', None)
        super().__init__(**kwargs)
        self.model1 = model1
        self.model2 = model2

    def grow(self) -> bool:
        logger.info('Starting EA-based active learning')
        logger.info(f'Random seed: {self.int_random_seed}')

        if self.target1 is not None and self.target2 is not None:
            if self.use_diversity:
                logger.info('Mode: Two-target QBC + Diversity (NSGA-II 3-obj)')
                self._ea_mode = 'two_target_div'
            else:
                logger.info('Mode: Two-target QBC only (NSGA-II 2-obj)')
                self._ea_mode = 'two_target_no_div'
        elif self.target is not None:
            logger.info('Mode: Single-target QBC + Diversity (single-objective EA)')
            self._ea_mode = 'single_target_div'
        else:
            logger.error('Must specify either (target1, target2) or target')
            return False

        return self._run_al_loop()

    def _select_next_batch(self, x_train, x_val, results_m1, results_m2, step_size):
        step = min(step_size, len(x_val))
        if self._ea_mode == 'two_target_no_div':
            diffs_df = self._create_quality_dataframe(results_m1, results_m2)
            return self._EA_select_batch_twoQuality_only(x_val, diffs_df, step)

        elif self._ea_mode == 'two_target_div':
            diffs_df = self._create_quality_dataframe(results_m1, results_m2)
            return self._EA_select_batch_twoQuality_plus_diversity(x_train, x_val, diffs_df, step)

        else:  # single_target_div
            diff = np.abs(results_m1[self.target]['model_pred'] - results_m2[self.target]['model_pred'])
            quality_series = pd.Series(self._min_max_standardize(diff), index=x_val.index)
            return self._EA_select_batch_singleObjective_with_diversity(x_train, x_val, quality_series, step)

    def _create_quality_dataframe(self, results_m1, results_m2):
        """Build quality DataFrame with columns 'q1'/'q2' from two-target disagreement."""
        diff1 = np.abs(results_m1[self.target1]['model_pred'] - results_m2[self.target1]['model_pred'])
        diff2 = np.abs(results_m1[self.target2]['model_pred'] - results_m2[self.target2]['model_pred'])
        index = getattr(results_m1[self.target1]['model_pred'], 'index', None)
        return pd.DataFrame({
            'q1': self._min_max_standardize(diff1),
            'q2': self._min_max_standardize(diff2),
        }, index=index)

    def _scale_and_normalize(self, trainingdata, candidates, do_scale=True, normalize_feats=True):
        """Scale and normalize training and candidate feature matrices."""
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
        return X_train, X_pool

    # ============================================================
    # EA selection methods
    # ============================================================

    def _EA_select_batch_twoQuality_only(self, candidates, diffs_df, step_size,
                                          pop_size=None, n_gen=None, p_cx=None,
                                          p_mut=None, random_state=None):
        """NSGA-II on [mean(q1), mean(q2)]."""
        pop_size = pop_size or self.pop_size
        n_gen = n_gen or self.n_gen
        p_cx = p_cx if p_cx is not None else self.p_cx
        p_mut = p_mut if p_mut is not None else self.p_mut
        random_state = random_state if random_state is not None else self.int_random_seed

        n_pool = len(candidates)
        diffs_df = diffs_df.reindex(candidates.index)
        q1 = np.nan_to_num(diffs_df["q1"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        q2 = np.nan_to_num(diffs_df["q2"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        def fitness(chrom):
            idx = np.asarray(chrom, dtype=int)
            return float(np.mean(q1[idx])), float(np.mean(q2[idx]))

        best = self._nsga2_evolve(fitness, n_pool, step_size, pop_size, n_gen, p_cx, p_mut, random_state)
        return candidates.index[np.asarray(best, dtype=int)].tolist() if best is not None else []

    def _EA_select_batch_twoQuality_plus_diversity(self, trainingdata, candidates, diffs_df,
                                                    step_size, do_scale=True, normalize_feats=True,
                                                    pop_size=None, n_gen=None, p_cx=None, p_mut=None,
                                                    random_state=None, w_train=None, w_within=None):
        """NSGA-II on [mean(q1), mean(q2), diversity]."""
        pop_size = pop_size or self.pop_size
        n_gen = n_gen or self.n_gen
        p_cx = p_cx if p_cx is not None else self.p_cx
        p_mut = p_mut if p_mut is not None else self.p_mut
        random_state = random_state if random_state is not None else self.int_random_seed
        w_train = w_train if w_train is not None else self.w_train
        w_within = w_within if w_within is not None else self.w_within

        X_train, X_pool = self._scale_and_normalize(trainingdata, candidates, do_scale, normalize_feats)
        n_pool = X_pool.shape[0]
        diffs_df = diffs_df.reindex(candidates.index)
        q1 = np.nan_to_num(diffs_df["q1"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        q2 = np.nan_to_num(diffs_df["q2"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sim_pt = (X_pool @ X_train.T).mean(axis=1).astype(np.float32) if X_train.shape[0] > 0 else np.zeros(n_pool, dtype=np.float32)
        G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)

        def fitness(chrom):
            idx = np.asarray(chrom, dtype=int)
            div_train = float(np.mean(1.0 - sim_pt[idx]))
            div_within = 1.0 if len(idx) <= 1 else float(1.0 - np.mean(G_pp[np.ix_(idx, idx)][~np.eye(len(idx), dtype=bool)]))
            return (float(np.mean(q1[idx])), float(np.mean(q2[idx])),
                    float(w_train * div_train + w_within * div_within))

        best = self._nsga2_evolve(fitness, n_pool, step_size, pop_size, n_gen, p_cx, p_mut, random_state)
        return candidates.index[np.asarray(best, dtype=int)].tolist() if best is not None else []

    def _EA_select_batch_singleObjective_with_diversity(self, trainingdata, candidates, q,
                                                         step_size, do_scale=True, normalize_feats=True,
                                                         pop_size=None, n_gen=None, p_cx=None, p_mut=None,
                                                         random_state=None, w_train=None, w_within=None,
                                                         lambda_div=None):
        """Maximize J(B) = mean(q[B]) + lambda_div * (w_train*div_train + w_within*div_within)."""
        pop_size = pop_size or self.pop_size
        n_gen = n_gen or self.n_gen
        p_cx = p_cx if p_cx is not None else self.p_cx
        p_mut = p_mut if p_mut is not None else self.p_mut
        random_state = random_state if random_state is not None else self.int_random_seed
        w_train = w_train if w_train is not None else self.w_train
        w_within = w_within if w_within is not None else self.w_within
        lambda_div = lambda_div if lambda_div is not None else self.lambda_div

        rng = np.random.default_rng(random_state)
        X_train, X_pool = self._scale_and_normalize(trainingdata, candidates, do_scale, normalize_feats)
        n_pool = X_pool.shape[0]
        B = int(min(step_size, n_pool))
        if B <= 0:
            return []

        qv = q.reindex(candidates.index).values.astype(np.float32) if isinstance(q, pd.Series) else np.asarray(q, dtype=np.float32).reshape(-1)
        qv = np.nan_to_num(qv, nan=0.0, posinf=0.0, neginf=0.0)
        sim_pt = (X_pool @ X_train.T).mean(axis=1).astype(np.float32) if X_train.shape[0] > 0 else np.zeros(n_pool, dtype=np.float32)
        G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)

        def _div(idx):
            div_train = float(np.mean(1.0 - sim_pt[idx]))
            div_within = 1.0 if len(idx) <= 1 else float(1.0 - np.mean(G_pp[np.ix_(idx, idx)][~np.eye(len(idx), dtype=bool)]))
            return w_train * div_train + w_within * div_within

        def objective(chrom):
            idx = np.asarray(chrom, dtype=int)
            return float(np.mean(qv[idx])) + lambda_div * _div(idx)

        pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

        for _ in range(n_gen):
            scores = np.array([objective(ind) for ind in pop], dtype=float)
            elite_k = max(2, int(0.2 * pop_size))
            elites = [pop[i] for i in np.argsort(scores)[::-1][:elite_k]]
            offspring = []
            while len(offspring) < (pop_size - elite_k):
                i1, i2 = rng.integers(0, pop_size), rng.integers(0, pop_size)
                p1 = pop[i1] if scores[i1] >= scores[i2] else pop[i2]
                j1, j2 = rng.integers(0, pop_size), rng.integers(0, pop_size)
                p2 = pop[j1] if scores[j1] >= scores[j2] else pop[j2]
                c1, c2 = p1.copy(), p2.copy()
                if rng.random() < p_cx:
                    c1, c2 = self._ordered_crossover(c1, c2, rng)
                c1 = self._make_unique(self._mutate_swap(self._make_unique(c1, n_pool, rng), n_pool, rng, p_mut), n_pool, rng)
                c2 = self._make_unique(self._mutate_swap(self._make_unique(c2, n_pool, rng), n_pool, rng, p_mut), n_pool, rng)
                offspring.append(c1)
                if len(offspring) < (pop_size - elite_k):
                    offspring.append(c2)
            pop = elites + offspring[: pop_size - elite_k]

        final_scores = np.array([objective(ind) for ind in pop], dtype=float)
        best_local = pop[int(np.argmax(final_scores))]
        return candidates.index[np.asarray(best_local, dtype=int)].tolist()
