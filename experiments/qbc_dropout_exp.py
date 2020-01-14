from functools import partial
from keras.callbacks import EarlyStopping

import numpy as np
from modAL.uncertainty import entropy_sampling

from datasets.mnist_ds import get_categorical_mnist
from models.mnist_models import get_qbc_model
from modAL.qbc_dropout import qbc_dropout_strategy
from modAL import KerasActiveLearner, DropoutActiveLearner
import experiments.al_experiment as exp


(x, y), (x_test, y_test) = get_categorical_mnist()

POOL_SIZE = 9500
INIT_SIZE = 2000
BATCH_SIZE = 10

n_labeled_examples = x.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)
x_train = x[training_indices]
y_train = y[training_indices]

preset_batch = partial(qbc_dropout_strategy, n_instances=BATCH_SIZE, cmt_size=100)
preset_entropy = partial(entropy_sampling, n_instances=BATCH_SIZE, proba=False)
es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)

for i in range(2, 3):
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)
    x_train = x[training_indices]
    y_train = y[training_indices]
    x_pool = np.delete(x, training_indices, axis=0)
    y_pool = np.delete(y, training_indices, axis=0)

    # model = get_qbc_model(mc=False)
    # entropy_learner = KerasActiveLearner(
    #     estimator=model,
    #     X_training=x_train,
    #     y_training=y_train,
    #     query_strategy=preset_entropy,
    #     validation_data=(x_test, y_test),
    #     epochs=20,
    #     callbacks=[es]
    # )
    #
    # entropy_exp = exp.Experiment(
    #     learner=entropy_learner,
    #     X_pool=x_pool,
    #     y_pool=y_pool,
    #     X_val=x_test,
    #     y_val=y_test,
    #     n_queries=1,
    #     random_seed=i,
    #     pool_size=POOL_SIZE,
    #     name='mnist_entropy_cmt100_' + str(i)
    # )
    #
    # entropy_exp.run()
    # print('entropy performance:', entropy_exp.performance_history)
    # print('entropy query time:', entropy_exp.time_per_query_history)
    # print('entropy fit time:', entropy_exp.time_per_fit_history)
    # entropy_exp.save_state('statistic/entropy_state_cmt40_' + str(i))

    mc_model = get_qbc_model(mc=True)
    qbc_learner = DropoutActiveLearner(
        estimator=mc_model,
        X_training=x_train,
        y_training=y_train,
        query_strategy=preset_batch,
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=[es]
    )

    qbc_dropout_exp = exp.Experiment(
        learner=qbc_learner,
        X_pool=x_pool,
        y_pool=y_pool,
        X_val=x_test,
        y_val=y_test,
        n_queries=1,
        random_seed=i,
        pool_size=POOL_SIZE,
        name='mnist_qbc_dropout_cmt100_' + str(i)
    )

    qbc_dropout_exp.run()
    print('qbc performance:', qbc_dropout_exp.performance_history)
    print('qbc query time:', qbc_dropout_exp.time_per_query_history)
    print('qbc fit time:', qbc_dropout_exp.time_per_fit_history)
    qbc_dropout_exp.save_state('statistic/qbc_state_cmt100_' + str(i))




