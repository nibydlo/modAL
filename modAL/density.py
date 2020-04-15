"""
Measures for estimating the information density of a given sample.
"""
from typing import Callable, Union, Tuple

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors._ball_tree import BallTree

from modAL.uncertainty import classifier_entropy, classifier_uncertainty, classifier_margin
from modAL.utils import multi_argmax
from modAL.utils.data import modALinput


def similarize_distance(distance_measure: Callable) -> Callable:
    """
    Takes a distance measure and converts it into a information_density measure.

    Args:
        distance_measure: The distance measure to be converted into information_density measure.

    Returns:
        The information_density measure obtained from the given distance measure.
    """

    def sim(*args, **kwargs):
        return 1 / (1 + distance_measure(*args, **kwargs))

    return sim


cosine_similarity = similarize_distance(cosine)
euclidean_similarity = similarize_distance(euclidean)


def information_density(X: modALinput, metric: Union[str, Callable] = 'euclidean') -> np.ndarray:
    """
    Calculates the information density metric of the given data using the given metric.

    Args:
        X: The data for which the information density is to be calculated.
        metric: The metric to be used. Should take two 1d numpy.ndarrays for argument.

    Todo:
        Should work with all possible modALinput.
        Perhaps refactor the module to use some stuff from sklearn.metrics.pairwise

    Returns:
        The information density for each sample.
    """
    # inf_density = np.zeros(shape=(X.shape[0],))
    # for X_idx, X_inst in enumerate(X):
    #     inf_density[X_idx] = sum(similarity_measure(X_inst, X_j) for X_j in X)
    #
    # return inf_density/X.shape[0]

    similarity_mtx = 1 / (1 + pairwise_distances(X, X, metric=metric))

    return similarity_mtx.mean(axis=1)


def classifier_modified_margin(classifier: BaseEstimator, X: modALinput, proba=True, **predict_kwargs) -> np.ndarray:
    return 1 - classifier_margin(classifier, X, proba)


uncertainty_measure_dict = {
    'least_confident': classifier_uncertainty,
    'margin': classifier_modified_margin,
    'entropy': classifier_entropy
}


def sud(classifier: BaseEstimator,
        X: modALinput,
        n_instances: int = 1,
        k_neighbours: int = 20,
        uncertainty_measure='entropy',
        transform=None,
        top_uncertain=1000,
        with_mult=True,
        sparse=False,
        **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    if uncertainty_measure not in uncertainty_measure_dict:
        raise ValueError('uncertainty measure can be equal only to "least_confident", "margin" or "entropy"')

    print('with_mult', with_mult)
    print('sparse', sparse)

    uncertainty = uncertainty_measure_dict[uncertainty_measure](classifier, X, **uncertainty_measure_kwargs)

    if transform is not None:
        X = transform(X)

    ball_tree = BallTree(X, leaf_size=5)

    top_uncertain_idx = multi_argmax(uncertainty, n_instances=top_uncertain)

    top_uncertain_density = \
        1 / np.array(
            [np.mean(
                ball_tree.query(x.reshape(1, -1), k=k_neighbours, return_distance=True)[0]
            ) for x in (X[top_uncertain_idx])]
        ) if not sparse else \
            np.array(
                [np.mean(
                    ball_tree.query(x.reshape(1, -1), k=k_neighbours, return_distance=True)[0]
                ) for x in (X[top_uncertain_idx])]
            )

    if with_mult:
        sud_measure = uncertainty[top_uncertain_idx] * top_uncertain_density
    else:
        sud_measure = top_uncertain_density

    query_idx = top_uncertain_idx[multi_argmax(sud_measure, n_instances=n_instances)]
    return query_idx, np.array([])
