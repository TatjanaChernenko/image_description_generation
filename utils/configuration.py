import json
import os

from .singleton import singleton


@singleton
class Configuration():
    __json = None

    def __init__(self):
        path = os.path.join(os.path.dirname(__file__),
                            '..', 'config.json')
        with open(path) as json_data:
            self.__json = json.load(json_data)

    def __getitem__(self, key):
        return self.__json[key]
