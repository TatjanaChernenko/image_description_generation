import os
import json
from typing import List

from .singleton import singleton


@singleton
class ModelSettings():
    defaults: dict
    model_settings: List[dict]

    def __init__(self):
        current_path = os.path.dirname(__file__)
        base = os.path.join(current_path, '..', 'assets', 'model_settings')
        with open(os.path.join(base, 'defaults.json')) as fp:
            self.defaults = json.load(fp)

        with open(os.path.join(base, 'settings.json')) as fp:
            self.model_settings = json.load(fp)

    def get(self, name: str = '', id: int = -1):
        setting = dict
        if name:
            setting = self.__get_by_name(name)
        elif id != -1:
            setting = self.__get_by_id(id)
        else:
            raise ValueError('Please enter either `name` or `id`')

        return {
            **setting,
            'hparameter': self.__add_defaults(setting, key='hparameter'),
            'flags': self.__add_defaults(setting, key='flags')
        }

    def __add_defaults(self, setting: dict, key: str):
        return {
            **self.defaults[key],
            **setting[key]
        }

    def __get_by_name(self, name):
        item = [setting for setting in self.model_settings
                if setting['name'] == name]
        if len(item) != 1:
            raise ValueError(
                f'Settings for model `{name}` not found or more than once')

        return item[0]

    def __get_by_id(self, id):
        return self.model_settings[id]
