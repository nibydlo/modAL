from typing import Union, Tuple

import numpy as np
from modAL.utils import multi_argmax

from modAL.models.base import BaseLearner, BaseCommittee
from sklearn.exceptions import NotFittedError

from modAL.utils.data import modALinput
from sklearn.base import BaseEstimator
import scipy.sparse as sp


def dropout_uncertainty(classifier: BaseEstimator, X: modALinput, cmt_size=10, **predict_proba_kwargs) -> np.ndarray:
    predictions = []
    for _ in range(cmt_size):
        try:
            predictions.append(classifier.predict(X, **predict_proba_kwargs))
        except NotFittedError:
            return np.ones(shape=(X.shape[0], ))

    max_means = []
    for idx in range(X.shape[0]):
        px = np.array([p[idx] for p in predictions])
        max_means.append(px.mean(axis=0).max())

    uncertainty = 1 - np.array(max_means)
    return uncertainty


def qbc_dropout_strategy(classifier: Union[BaseLearner, BaseCommittee],
                       X: Union[np.ndarray, sp.csr_matrix],
                       n_instances: int = 20,
                       **dropout_uncertainty_kwargs
                       ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    uncertainty = dropout_uncertainty(classifier, X, **dropout_uncertainty_kwargs)
    query_idx = multi_argmax(uncertainty, n_instances=n_instances)
    return query_idx, X[query_idx]