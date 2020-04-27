from typing import Union, Tuple

import numpy as np
from modAL.utils import multi_argmax

from modAL.models.base import BaseLearner, BaseCommittee

import scipy.sparse as sp


def learning_loss_strategy(classifier: Union[BaseLearner, BaseCommittee],
                           X: Union[np.ndarray, sp.csr_matrix],
                           n_instances: int = 20,
                           **predict_kwargs
                           ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    losses = classifier.estimator.predict_loss(X, **predict_kwargs)
    query_idx = multi_argmax(losses, n_instances=n_instances).squeeze(axis=1)
    return query_idx, np.array([])

