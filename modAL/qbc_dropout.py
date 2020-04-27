from typing import Union, Tuple
import math

import numpy as np
from modAL.utils import multi_argmax

from modAL.models.base import BaseLearner, BaseCommittee
from sklearn.exceptions import NotFittedError

from modAL.utils.data import modALinput
from sklearn.base import BaseEstimator
import scipy.sparse as sp
from scipy.stats import entropy


def get_least_confidence(predictions):
    return 1 - np.max(predictions, axis=1)


def get_margin(predictions):
    part = np.partition(-predictions, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]
    return -margin


def get_entropy(predictions):
    return np.transpose(entropy(np.transpose(predictions)))


uncertainty_measure_dict = {
    'least_confident': get_least_confidence,
    'margin': get_margin,
    'entropy': get_entropy
}


def predict_by_committee(classifier: BaseEstimator, X: modALinput, cmt_size=10, **predict_proba_kwargs) -> np.ndarray:
    predictions = []
    for _ in range(cmt_size):
        try:
            predictions.append(classifier.predict_proba(X, with_dropout=True, **predict_proba_kwargs))
        except NotFittedError:
            return np.ones(shape=(X.shape[0],))

    return predictions


def qbc_uncertainty_sampling(
        classifier: Union[BaseLearner, BaseCommittee],
        X: Union[np.ndarray, sp.csr_matrix],
        n_instances: int = 20,
        cmt_size: int = 10,
        uncertainty_measure='entropy',
        **dropout_uncertainty_kwargs
) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix, list]]:

    if uncertainty_measure not in uncertainty_measure_dict:
        raise ValueError('uncertainty measure can be equal only to "least_confident", "margin" or "entropy"')

    committee_predictions = predict_by_committee(
        classifier=classifier,
        X=X,
        cmt_size=cmt_size,
        **dropout_uncertainty_kwargs
    )
    uncertainty = uncertainty_measure_dict[uncertainty_measure](np.mean(committee_predictions, axis=0))

    query_idx = multi_argmax(uncertainty, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        new_batch = [x[query_idx] for x in X]
    else:
        new_batch = X[query_idx]

    return query_idx, new_batch


dis_func = np.vectorize(lambda x: 0 if x == 0 else x * math.log(x))


def get_disagreement(committee_predictions, committee_size):
    # 0 for members of committee, -1 for classes
    disagreement = np.sum(dis_func(committee_predictions), axis=(0, -1)) / committee_size
    return disagreement


def bald_sampling(
        classifier: Union[BaseLearner, BaseCommittee],
        X: Union[np.ndarray, sp.csr_matrix],
        n_instances: int = 20,
        cmt_size: int = 10,
        uncertainty_measure='entropy',
        **dropout_uncertainty_kwargs
) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix, list]]:

    if uncertainty_measure not in uncertainty_measure_dict:
        raise ValueError('uncertainty measure can be equal only to "least_confident", "margin" or "entropy"')

    committee_predictions = predict_by_committee(
        classifier=classifier,
        X=X,
        cmt_size=cmt_size,
        **dropout_uncertainty_kwargs
    )
    uncertainty = uncertainty_measure_dict[uncertainty_measure](np.mean(committee_predictions, axis=0))

    disagreement = get_disagreement(committee_predictions, cmt_size)
    query_idx = multi_argmax(uncertainty + disagreement, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        new_batch = [x[query_idx] for x in X]
    else:
        new_batch = X[query_idx]

    return query_idx, new_batch


def bald_modal_sampling(
        classifier: Union[BaseLearner, BaseCommittee],
        X: Union[np.ndarray, sp.csr_matrix],
        n_instances: int = 20,
        with_dropout=True,
        **dropout_uncertainty_kwargs
) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix, list]]:

    img_predictions = classifier.predict_proba([X[0], np.zeros_like(X[1])], with_dropout=with_dropout)
    txt_predictions = classifier.predict_proba([np.zeros_like(X[0]), X[1]], with_dropout=with_dropout)
    both_predictions = classifier.predict_proba(X, with_dropout=with_dropout)

    committee_predictions = [img_predictions, txt_predictions, both_predictions]
    uncertainty = get_entropy(np.mean(committee_predictions, axis=0))
    disagreement = get_disagreement(committee_predictions, 3)

    query_idx = multi_argmax(uncertainty + disagreement, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        new_batch = [x[query_idx] for x in X]
    else:
        new_batch = X[query_idx]

    return query_idx, new_batch


def bald_trident_sampling(
    classifier: Union[BaseLearner, BaseCommittee],
        X: Union[np.ndarray, sp.csr_matrix],
        n_instances: int = 20,
        with_dropout=False,
        cmt_size=1,
        **dropout_uncertainty_kwargs
) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix, list]]:

    committee_predictions = []
    for i in range(cmt_size):
        common_predictions, img_predictions, txt_predictions = classifier.predict_proba(X, with_dropout=with_dropout)
        committee_predictions.append(common_predictions)
        committee_predictions.append(img_predictions)
        committee_predictions.append(txt_predictions)

    uncertainty = get_entropy(np.mean(committee_predictions, axis=0))
    disagreement = get_disagreement(committee_predictions, cmt_size * 3)
    query_idx = multi_argmax(uncertainty + disagreement, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        new_batch = [x[query_idx] for x in X]
    else:
        new_batch = X[query_idx]

    return query_idx, new_batch


def bald_trident_based_sampling(
    classifier: Union[BaseLearner, BaseCommittee],
        X: Union[np.ndarray, sp.csr_matrix],
        cmt_size=3,
        n_instances: int = 20
) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix, list]]:

    committee_predictions = []
    for _ in range(cmt_size):
        committee_predictions.append(classifier.predict_proba(X, with_dropout=True)[0])

    uncertainty = get_entropy(np.mean(committee_predictions, axis=0))
    disagreement = get_disagreement(committee_predictions, 3)
    query_idx = multi_argmax(uncertainty + disagreement, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        new_batch = [x[query_idx] for x in X]
    else:
        new_batch = X[query_idx]

    return query_idx, new_batch