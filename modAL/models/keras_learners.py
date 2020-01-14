from typing import Any

from modAL.models.base import BaseLearner

from modAL.models.learners import ActiveLearner
from modAL.utils.data import modALinput
from sklearn.metrics import accuracy_score

import numpy as np
import tensorflow.python.keras as keras


class KerasActiveLearner(ActiveLearner):
    def score(self, X: modALinput, y: modALinput, **score_kwargs) -> Any:
        return self.estimator.evaluate(X, y, verbose=0)[1]


class DropoutActiveLearner(ActiveLearner):
    def score(self, X: modALinput, y: modALinput, **score_kwargs) -> Any:
        mc_predictions = []
        for i in range(10):
            y_p = self.estimator.predict(X, batch_size=1000)
            mc_predictions.append(y_p)
        accs = []
        for y_p in mc_predictions:
            acc = accuracy_score(y.argmax(axis=1), y_p.argmax(axis=1))
            accs.append(acc)
        return sum(accs) / len(accs)


class LearningLossActiveLearner(ActiveLearner):
    def __init__(self, loss_estimator, **al_init_kwargs):
        self.loss_estimator = loss_estimator
        super().__init__(**al_init_kwargs)

    def score(self, X: modALinput, y: modALinput, **score_kwargs) -> Any:
        return self.estimator.evaluate(X, y, verbose=0)[1]

    def predict_loss(self, X: modALinput, **predict_kwargs) -> Any:
        return self.loss_estimator.predict(X, **predict_kwargs)

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        if not bootstrap:
            self.estimator.fit(self.X_training, self.y_training, **fit_kwargs)
            y_predicted = self.estimator.predict(self.X_training)
            losses = keras.losses.mean_squared_error(self.y_training, y_predicted)
            self.loss_estimator.fit(self.X_training, losses)
        else:
            n_instances = self.X_training.shape[0]
            bootstrap_idx = np.random.choice(range(n_instances), n_instances, replace=True)
            self.estimator.fit(self.X_training[bootstrap_idx], self.y_training[bootstrap_idx], **fit_kwargs)
            y_predicted = self.estimator.predict(self.X_training[bootstrap_idx])
            losses = keras.losses.mean_squared_error(self.y_training[bootstrap_idx], y_predicted)
            self.loss_estimator.fit(self.X_training[bootstrap_idx], losses)

        return self

    def _fit_on_new(self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        if not bootstrap:
            self.estimator.fit(X, y, **fit_kwargs)

            y_predicted = self.estimator.predict(X)
            losses = keras.losses.mean_squared_error(y, y_predicted)
            self.loss_estimator.fit(X, losses)
        else:
            bootstrap_idx = np.random.choice(range(X.shape[0]), X.shape[0], replace=True)
            self.estimator.fit(X[bootstrap_idx], y[bootstrap_idx])

            y_predicted = self.estimator.predict(X[bootstrap_idx])
            losses = keras.losses.mean_squared_error(y[bootstrap_idx], y_predicted)
            self.loss_estimator.fit(X[bootstrap_idx], losses)

        return self
