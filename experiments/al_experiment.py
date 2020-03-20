from typing import Union
from pathlib import Path

from sklearn.utils import shuffle

import random
import numpy as np
import scipy.sparse as sp
import pickle
import time
import logging

from modAL import LearningLossActiveLearner
from modAL.models.base import BaseLearner, BaseCommittee

random_name_length = 5


def get_random_name():
    return ''.join(map(str, np.random.randint(low = 0, high = 9, size = random_name_length)))

def is_multimodal(X):
    return isinstance(X, list) and isinstance(X[0], np.ndarray)

class Experiment:

    def __init__(
            self,
            learner: Union[BaseLearner, BaseCommittee],
            X_pool: Union[np.ndarray, sp.csr_matrix, list],
            y_pool: Union[np.ndarray, sp.csr_matrix],
            X_val: Union[np.ndarray, sp.csr_matrix, list] = None,
            y_val: Union[np.ndarray, sp.csr_matrix] = None,
            n_instances: int = 1,
            n_queries: int = 10,
            random_seed: int = random.randint(0, 100),
            pool_size: int = -1,
            name: str = get_random_name(),
            **teach_kwargs
    ):
        self.learner = learner
        self.n_queries = n_queries
        self.random_seed = random_seed
        self.n_instances = n_instances
        # self.init_size = self.learner.X_training.shape[0]

        if is_multimodal(X_pool):
            n_instances = X_pool[0].shape[0]
            idx = np.random.choice(range(n_instances), n_instances, replace=False)
            X_pool = [x[idx] for x in X_pool]
            y_pool = y_pool[idx]
        else:
            X_pool, y_pool = shuffle(X_pool, y_pool, random_state=random_seed)

        real_size = X_pool[0].shape[0] if is_multimodal(X_pool) else X_pool.shape[0]

        if 0 <= pool_size < real_size:
            if isinstance(X_pool, list) and isinstance(X_pool[0], np.ndarray):
                X_pool = [x[:pool_size] for x in X_pool]
            else:
                X_pool = X_pool[:pool_size]
            y_pool = y_pool[:pool_size]
        else:
            pool_size = real_size

        self.pool_size = pool_size
        self.X_pool = X_pool
        self.y_pool = y_pool

        self.X_val = X_pool if X_val is None else X_val
        self.y_val = y_pool if y_val is None else y_val

        self.performance_history = []
        self.time_per_query_history = []
        self.time_per_fit_history = []

        self.name = name
        self._setup_logger()
        self.teach_kwargs = teach_kwargs

    def _setup_logger(self):
        self.logger = logging.getLogger('exp_' + self.name)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('log/exp_' + self.name + '.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _out_of_data_warn(self):

        self.logger.warning('pool does not have enough data, batch size = '
                           + str(self.n_instances)
                           + ' but pool size = '
                           + str(self.pool_size))

    def save_state(self, state_name):
        state = {
            # 'init_size' : self.init_size,
            'n_instances' : self.n_instances,
            'n_queries' : self.n_queries,
            'performance_history' : self.performance_history,
            'time_per_query_history' : self.time_per_query_history,
            'time_per_fit_history' : self.time_per_fit_history
        }
        if isinstance(self.learner, LearningLossActiveLearner):
            state['loss_history'] = self.learner.loss_history
            state['learning_loss_history'] = self.learner.learning_loss_history
        Path('/'.join(state_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        with open(state_name + '.pkl', 'wb') as f:
            pickle.dump(state, f)

    def step(self, i=-1):
        self.logger.info('start step #' + str(i))
        start_time_step = time.time()
        if self.n_instances > self.pool_size:
            self._out_of_data_warn()
            return
        start_time_query = time.time()

        query_index, query_instance = self.learner.query(self.X_pool)
        self.logger.info('query idx: ' + str(query_index))
        self.time_per_query_history.append(time.time() - start_time_query)

        if is_multimodal(self.X_pool):
            X = [x[query_index] for x in self.X_pool]
        else:
            X = self.X_pool[query_index]
        y = self.y_pool[query_index]

        start_time_fit = time.time()
        self.learner.teach(X=X, y=y, **self.teach_kwargs)
        self.time_per_fit_history.append(time.time() - start_time_fit)

        if is_multimodal(self.X_pool):
            self.X_pool = [np.delete(x, query_index, axis=0) for x in self.X_pool]
        else:
            self.X_pool = np.delete(self.X_pool, query_index, axis=0)
        self.y_pool = np.delete(self.y_pool, query_index, axis=0)

        score = self.learner.score(self.X_val, self.y_val)
        self.performance_history.append(score)
        self.logger.info('finish step #' + str(i) + ' for ' + str(time.time() - start_time_step) + ' sec')
        self.logger.info('current val_accuracy: ' + str(score))
        return query_index, query_instance, score

    def run(self):
        self.logger.info('start experiment process')
        start_time = time.time()
        score = self.learner.score(self.X_val, self.y_val)
        self.performance_history.append(score)
        self.logger.info('initial val_accuracy: ' + str(score))
        for i in range(self.n_queries):
            if self.n_instances > self.pool_size:
                self._out_of_data_warn()
                break
            self.step(i)
        self.logger.info('finish experiment process for ' + str(time.time() - start_time) + ' sec')
        return self.performance_history
