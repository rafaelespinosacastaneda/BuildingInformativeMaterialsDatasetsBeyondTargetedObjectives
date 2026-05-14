from .active_learning_grower import ActiveLearningGrower
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiObjPolicy(ActiveLearningGrower):
    """
    Multi-objective active learning strategy.

    Selects samples by combining model disagreement across properties.
    - target1 + target2: combines those two properties
    - target: single-property disagreement
    - neither: combines all three properties
    """

    def __init__(self, model1, model2, **kwargs):
        self.target1 = kwargs.pop('target1', None)
        self.target2 = kwargs.pop('target2', None)
        self.target = kwargs.pop('target', None)
        super().__init__(**kwargs)
        self.model1 = model1
        self.model2 = model2

    def grow(self) -> bool:
        logger.info('Starting Multi-objective active learning')
        logger.info(f'Random seed: {self.int_random_seed}')
        return self._run_al_loop()

    def _select_next_batch(self, x_train, x_val, results_model1, results_model2, step_size):
        if self.target1 is not None and self.target2 is not None:
            combined_diff = self._combine_normalized_differences(
                results_model1, results_model2, self.target1, self.target2
            )
        elif self.target is not None:
            diff = np.abs(results_model1[self.target]['model_pred'] - results_model2[self.target]['model_pred'])
            combined_diff = self._min_max_standardize(diff)
        else:
            combined_diff = self._combine_all_properties_differences(results_model1, results_model2)

        return pd.Series(combined_diff, index=x_val.index).nlargest(step_size).index.tolist()
