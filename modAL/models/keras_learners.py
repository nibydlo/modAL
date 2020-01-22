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
    def __init__(self, cmt_size=10, **al_init_kwargs):
        self.cmt_size=cmt_size
        super().__init__(**al_init_kwargs)

    def score(self, X: modALinput, y: modALinput, **score_kwargs) -> Any:
        mc_predictions = []
        for i in range(self.cmt_size):
            y_p = self.estimator.predict(X, batch_size=1000)
            mc_predictions.append(y_p)
        predictions = np.mean(mc_predictions, axis=0)
        return accuracy_score(y.argmax(axis=1), predictions.argmax(axis=1))


class LearningLossActiveLearner(ActiveLearner):

    def __init__(self, **al_init_kwargs):
        self.loss_performance = []
        super().__init__(**al_init_kwargs)

    def score(self, X: modALinput, y: modALinput, **score_kwargs) -> Any:
        eval_res = self.estimator.evaluate(X, [y, np.zeros((len(y), 1))])  # evaluate will return [loss, target_loss, loss_loss, target_acc, loss_ass
        return eval_res[3]

    def predict_loss(self, X: modALinput, **predict_kwargs) -> Any:
        _, loss_pred = self.estimator.predict(X, **predict_kwargs)  # first will be target prediction

        return loss_pred

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        old_score = -1
        new_score = self.score(self.X_training, self.y_training)
        while new_score > old_score + 0.005:
            if not bootstrap:
                target_pred, loss_pred = self.estimator.predict(self.X_training)
                loss_true = keras.losses.mean_squared_error(self.y_training, target_pred)
                self.loss_performance.append(self.learning_loss_fun(loss_true, loss_pred))
                self.estimator.fit(self.X_training, [self.y_training, loss_true])
            else:
                n_instances = self.X_training.shape[0]
                bootstrap_idx = np.random.choice(range(n_instances), n_instances, replace=True)
                target_pred = self.estimator.predict(self.X_training[bootstrap_idx])
                loss_true = keras.losses.mean_squared_error(self.y_training[bootstrap_idx], target_pred)
                self.loss_estimator.fit(self.X_training[bootstrap_idx], [self.y_training[bootstrap_idx], loss_true])
            old_score = new_score
            new_score = self.score(self.X_training, self.y_training)
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
