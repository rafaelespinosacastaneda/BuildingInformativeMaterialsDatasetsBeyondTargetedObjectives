from .active_learning_grower import ActiveLearningGrower
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class QBCPolicy(ActiveLearningGrower):
    """
    Query-by-Committee active learning strategy.

    Selects samples where two models disagree most on the target property.
    """

    def __init__(self, model1, model2, target: str, **kwargs):
        super().__init__(**kwargs)
        self.model1 = model1
        self.model2 = model2
        self.target = target

    def grow(self) -> bool:
        logger.info('Starting QBC (Query-by-Committee) active learning')
        logger.info(f'Random seed: {self.int_random_seed}')
        logger.info(f'Target property: {self.target}')
        return self._run_al_loop()

    def _select_next_batch(self, x_train, x_val, results_model1, results_model2, step_size):
        target_prop = self.target_map.get(self.target, self.target)
        if target_prop not in results_model1 or target_prop not in results_model2:
            logger.warning(f'Target property {target_prop} not found in results')
            return []

        pred1 = np.asarray(results_model1[target_prop]['model_pred'])
        pred2 = np.asarray(results_model2[target_prop]['model_pred'])
        disagreement = pd.Series(np.abs(pred1 - pred2), index=x_val.index)
        return disagreement.nlargest(step_size).index.tolist()
