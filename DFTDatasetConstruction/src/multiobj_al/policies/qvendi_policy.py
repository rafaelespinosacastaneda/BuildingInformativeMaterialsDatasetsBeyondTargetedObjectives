from .active_learning_grower import ActiveLearningGrower, Bestindices_qVendiScoreBatch
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class QVendiPolicy(ActiveLearningGrower):
    """
    Q-Vendi active learning: combines model disagreement (quality) with diversity.
    - target1 + target2: combines those two properties (normalized diffs combined)
    - target: single-property disagreement (raw absolute diff, no normalization)
    - neither: combines all three properties (normalized diffs combined)
    """

    def __init__(self, model1, model2, **kwargs):
        self.target1 = kwargs.pop('target1', None)
        self.target2 = kwargs.pop('target2', None)
        self.target = kwargs.pop('target', None)
        super().__init__(**kwargs)
        self.model1 = model1
        self.model2 = model2

    def grow(self) -> bool:
        logger.info('Starting Q-Vendi active learning')
        logger.info(f'Random seed: {self.int_random_seed}')
        return self._run_al_loop()

    def _select_next_batch(self, x_train, x_val, results_model1, results_model2, step_size):
        if self.target1 is not None and self.target2 is not None:
            combined_diff = self._combine_normalized_differences(
                results_model1, results_model2, self.target1, self.target2
            )
        elif self.target is not None:
            combined_diff = np.abs(
                results_model1[self.target]['model_pred'] - results_model2[self.target]['model_pred']
            )
        else:
            combined_diff = self._combine_all_properties_differences(results_model1, results_model2)

        diffs = pd.Series(combined_diff, index=x_val.index)
        return Bestindices_qVendiScoreBatch(x_train, x_val, diffs, step_size)
