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
    losses = classifier.predict_loss(X, **predict_kwargs)
    # print('losses:', losses)
    query_idx = multi_argmax(losses, n_instances=n_instances).squeeze(axis=1)
    # print('idx:', query_idx, query_idx.shape)
    return query_idx, X[query_idx]

