import pickle

from .persistance import Persistance


class Binary(Persistance):
    @staticmethod
    def save(data, path):
        with open(path, 'wb') as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)


