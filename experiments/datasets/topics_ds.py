from importlib import resources as pkg_resources
from pickle import UnpicklingError
import json
import pandas as pd
import numpy as np
import pickle
from struct import unpack
from base64 import b64decode
from functools import partial
from keras.utils.np_utils import to_categorical

from experiments import config

IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50


def unpck(l, x):
    return unpack('%df' % l, b64decode(x.encode('utf-8')))


unpck_img = partial(unpck, IMG_LEN)
unpck_txt = partial(unpck, TXT_LEN)

ds_config = json.load(pkg_resources.open_binary(config, 'config.json'))
ds_path = ds_config['ds_path']
unpacked_ds_path = ds_config['unpacked_ds_path']

def get_unpacked_data():
    try:
        topics_pickle_in = open(unpacked_ds_path, 'rb')
        x_img, x_txt, y = pickle.load(topics_pickle_in)
    except (FileNotFoundError, UnpicklingError):
        df = pd.read_json(open(ds_path, 'rb'), lines=True)
        x_img = np.stack(df['x1'].map(unpck_img), axis=0)
        x_txt = np.stack(df['x2'].map(unpck_txt), axis=0)
        y = to_categorical(np.array(df['y1']), N_CLASSES)

        topics_pickle_out = open(unpacked_ds_path, "wb")
        pickle.dump((x_img, x_txt, y), topics_pickle_out)
        topics_pickle_out.close()

    return x_img, x_txt, y
