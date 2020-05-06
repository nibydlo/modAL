from typing import Union, Tuple

import numpy as np

import torch
import torch.nn.functional as F

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


def learning_loss_ideal(classifier: Union[BaseLearner, BaseCommittee],
                           X: Union[np.ndarray, sp.csr_matrix],
                           y,
                           n_instances: int = 20,
                           **predict_kwargs
                           ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    prediction = torch.tensor(classifier.estimator.predict(X))
    actual = torch.argmax(torch.tensor(y), dim=1)
    losses = F.nll_loss(prediction, actual, reduction='none')
    query_idx = multi_argmax(losses.detach().numpy(), n_instances=n_instances)

    return query_idx, np.array([])