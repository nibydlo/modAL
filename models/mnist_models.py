from tensorflow.python.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras import Model
import tensorflow.python.keras as keras
import numpy as np
import tensorflow as tf

num_classes = 10
mnist_input_shape = (28, 28, 1)

def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

def get_qbc_model(mc=False):
    inp = Input(mnist_input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inp)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = get_dropout(x, p=0.5, mc=mc)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


eps = 0.001

def get_learning_loss_model(batch_size=10):
    assert batch_size % 2 == 0, 'batch size should be even'

    def learning_loss_fun(_, y_predicted):
        @tf.function
        def kostyl_sign(x):
            try:
                return x / abs(x)
            except:
                return 0

        @tf.function
        def loss(x):
            a, b = x[0], x[1]
            t = kostyl_sign(a - b) * (a - b) + eps
            t = (t + abs(t)) / 2.0
            return t

        b = y_predicted[-batch_size:]
        # tf.print('b = ', b)
        # tf.print('b shape', tf.shape(b))
        # tf.print('reshaped:', tf.reshape(b, [-1, 2]))
        # tf.print('mapped:', tf.map_fn(loss, tf.reshape(b, [-1, 2])))
        res = tf.reduce_sum(tf.map_fn(loss, tf.reshape(b, [-1, 2])))

        # tf.print('loss from inside = ', res)
        tf.identity(res)
        # return sum([loss(b[i], b[i + 1]) for i in range(0, batch_size, 2)]) / batch_size
        return res

    inp = Input(mnist_input_shape)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(inp)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    y1 = keras.layers.GlobalAveragePooling2D()(x)
    y1 = Flatten()(y1)
    y1 = Dense(128, activation='relu')(y1)

    x = Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu')(inp)
    x = Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    y2 = keras.layers.GlobalAveragePooling2D()(x)
    y2 = Flatten()(y2)
    y2 = Dense(128, activation='relu')(y2)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)

    y3 = Dense(128, activation='relu')(x)
    y = keras.layers.concatenate([y1, y2, y3])
    y = Dense(128, activation='relu')(y)

    out1 = Dense(num_classes, activation='softmax', name='target_output')(x)
    out2 = Dense(1, name='loss_output')(y)
    # print(out2.shape)

    model1 = Model(inputs=inp, outputs=out1)
    model1.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model2 = Model(inputs=inp, outputs=out2)
    model2.compile(loss=learning_loss_fun,
                   optimizer='adam')

    return model1, model2
