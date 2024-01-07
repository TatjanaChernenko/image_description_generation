from collections import defaultdict
from itertools import chain
from typing import Dict, List, Set

from utils import Configuration


class Flatten():
    word_collection: Dict[str, list] = defaultdict(list)
    key = 'none'

    def __init__(self, properties: List[str]):
        self.properties = properties
        self.word_collection['coco'].append('eos')
        self.word_collection['coco'].append('--')
        self.word_collection['coco-a'].append('eos')
        self.word_collection['coco-a'].append('--')

    def __check_dict(self, dic, depth=0):
        result = []
        sorted_dict = dict(sorted(dic.items()))
        for key, value in sorted_dict.items():
            if key not in self.properties[self.key]:
                continue
            if value is None:
                result.append('--')
            else:
                result.append(self.__walk(value, depth=depth))
        return result if len(result) != 0 else None

    def __walk(self, data, depth=0):
        if isinstance(data, list):
            return [self.__walk(itm, depth=depth + 1) for itm in data]
        elif isinstance(data, dict):
            return self.__check_dict(data, depth=depth + 1)
        elif isinstance(data, str) or \
                isinstance(data, float) or isinstance(data, int):
            self.word_collection[self.key].append(data)
            return data

    def __make_list_flat(self, data, result=[]):
        if isinstance(data, str) or \
                isinstance(data, float) or isinstance(data, int):
            result.append(data)
        elif isinstance(data, list):
            for item in data:
                self.__make_list_flat(item, result)

    def __call__(self, data, key='none'):
        """
        Flattens the data for learning purposes
        """
        self.key = key
        walked_data = self.__walk(data)
        flat_data = []
        self.__make_list_flat(walked_data, result=flat_data)
        return flat_data
