import pickle

import keras.models


def set_file(name, data_to_save):
    with open(name, "wb") as save:
        pickle.dump(data_to_save, save)


def get_file(name):
    with open(name, "rb") as save:
        my_file = pickle.load(save)
        return my_file


def set_model(name, model):
    model.save(name)


def get_model(name):
    return keras.models.load_model(name)
