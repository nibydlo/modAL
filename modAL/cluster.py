from typing import Tuple
import numpy as np
from modAL.uncertainty import classifier_entropy

from modAL.utils.data import modALinput
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans


def cluster_sampling(classifier: BaseEstimator, X: modALinput,
                     n_instances: int = 1, transform=None,
                     **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    entropy = classifier_entropy(classifier, X, **uncertainty_measure_kwargs)

    if transform is not None:
        X_orig = X
        X = transform(X)

    km = KMeans(n_clusters=n_instances)
    km.fit(X)

    batch = []

    for label in range(n_instances):
        idx = np.where(km.labels_ == label)[0]
        max_entropy = 0
        max_i = 0
        for i in idx:
            if entropy[i] > max_entropy:
                max_entropy = entropy[i]
                max_i = i
        batch.append(max_i)
    query_idx = np.array(batch)
    return query_idx, (X if transform is None else X_orig)[query_idx]