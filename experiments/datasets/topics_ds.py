from importlib import resources as pkg_resources
from pickle import UnpicklingError

import pandas as pd
import numpy as np
import pickle
from struct import unpack
from base64 import b64decode
from functools import partial
from keras.utils.np_utils import to_categorical
from .resources import topics

IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50


def unpck(l, x):
    return unpack('%df' % l, b64decode(x.encode('utf-8')))


unpck_img = partial(unpck, IMG_LEN)
unpck_txt = partial(unpck, TXT_LEN)


def get_unpacked_data():
    try:
        topics_pickle_in = pkg_resources.open_binary(topics, 'unpacked_topics.pickle')
        x_img, x_txt, y = pickle.load(topics_pickle_in)
    except (FileNotFoundError, UnpicklingError):
        df = pd.read_json(pkg_resources.open_binary(topics, 'topics_dataset.json'), lines=True)

        x_img = np.stack(df['x1'].map(unpck_img), axis=0)
        x_txt = np.stack(df['x2'].map(unpck_txt), axis=0)
        y = to_categorical(np.array(df['y1']), N_CLASSES)

        topics_pickle_out = open("experiments/datasets/resources/topics/unpacked_topics.pickle", "wb")
        pickle.dump((x_img, x_txt, y), topics_pickle_out)
        topics_pickle_out.close()

    return x_img, x_txt, y
