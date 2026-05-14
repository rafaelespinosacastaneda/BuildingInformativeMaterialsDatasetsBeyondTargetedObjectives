from .active_learning_grower import ActiveLearningGrower
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RandomPolicy(ActiveLearningGrower):
    """Random sampling active learning strategy."""

    def __init__(self, model1, model2, **kwargs):
        super().__init__(**kwargs)
        self.model1 = model1
        self.model2 = model2

    def grow(self) -> bool:
        logger.info('Starting Random Sampling active learning')
        logger.info(f'Random seed: {self.int_random_seed}')
        return self._run_al_loop()

    def _select_next_batch(self, x_train, x_val, results_model1, results_model2, step_size):
        return np.random.choice(
            x_val.index, min(step_size, len(x_val)), replace=False
        ).tolist()
