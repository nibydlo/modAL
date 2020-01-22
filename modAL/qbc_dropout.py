from typing import Union, Tuple

import numpy as np
from modAL.utils import multi_argmax

from modAL.models.base import BaseLearner, BaseCommittee
from sklearn.exceptions import NotFittedError

from modAL.utils.data import modALinput
from sklearn.base import BaseEstimator
import scipy.sparse as sp
from scipy.stats import entropy


def dropout_uncertainty(classifier: BaseEstimator, X: modALinput, cmt_size=10, **predict_proba_kwargs) -> np.ndarray:
    predictions = []
    for _ in range(cmt_size):
        try:
            predictions.append(classifier.predict(X, **predict_proba_kwargs))
        except NotFittedError:
            return np.ones(shape=(X.shape[0], ))

    mean_predictions = np.mean(predictions, axis=0)
    return np.transpose(entropy(np.transpose(mean_predictions)))


def qbc_dropout_strategy(classifier: Union[BaseLearner, BaseCommittee],
                       X: Union[np.ndarray, sp.csr_matrix],
                       n_instances: int = 20,
                       **dropout_uncertainty_kwargs
                       ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    uncertainty = dropout_uncertainty(classifier, X, **dropout_uncertainty_kwargs)
    query_idx = multi_argmax(uncertainty, n_instances=n_instances)
    return query_idx, X[query_idx]