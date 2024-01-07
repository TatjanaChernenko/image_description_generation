from itertools import chain, groupby, filterfalse
from typing import Dict, List
from collections import defaultdict

from .flatten import Flatten


class Format():
    def __init__(self, properties: List[str]):
        self.properties = properties
        self.flatten = Flatten(properties=properties)

    def __fill(self, lst, length, filler='--'):
        if len(lst) != 0 and isinstance(lst[0], list):
            return lst + [[filler] * len(lst[0])] * (length - len(lst))
        return lst + [filler] * (length - len(lst))

    def __determine_max_length(self, list_of_dicts: List[dict]):
        length: Dict[str, list] = defaultdict(list)

        for item in list_of_dicts:
            for key, lst in item.items():
                if isinstance(lst, list):
                    if len(lst) != 0 and isinstance(lst[0], list):
                        max_length_sub = max([len(itm) for itm in lst])
                        length[key + '_sub'].append(max_length_sub)
                    length[key].append(len(lst))

        max_length = {
            key: max(lst)
            for key, lst in length.items()
        }

        return max_length

    def __normalize_length(self, list_of_dicts: List[dict]):
        max_length = self.__determine_max_length(list_of_dicts)
        result = []
        for item in list_of_dicts:
            transformed_item = {}
            for key, lst in item.items():
                if key + '_sub' in max_length:
                    sub_filled = [
                        self.__fill(lst_of_lst, max_length[key + '_sub'])
                        for lst_of_lst in lst
                    ]

                    transformed_item[key] = self.__fill(
                        sub_filled, max_length[key])
                else:
                    transformed_item[key] = self.__fill(lst, max_length[key]) \
                        if isinstance(lst, list) else lst
            result.append(transformed_item)
        return result

    def __restructure(self, data: dict):
        result_data = {}
        for key, value in data.items():
            if key == 'coco':
                result_data[key] = [{
                    'image_id': item['image']['id'],
                    'captions': self.flatten(item['captions'], key=key),
                    'categories': self.flatten(item['anns'], key=key)
                } for item in value]
            elif key == 'coco-a':
                tmp = [{
                    'image_id': item['image_id'],
                    'interaction': self.flatten(item, key=key),
                } for item in value]

                collected = defaultdict(list)
                grouped = groupby(tmp, lambda inter: inter['image_id'])
                for image_id, group in grouped:
                    collected[image_id].append([interaction['interaction']
                                                for interaction in group])

                result_data[key] = [{
                    'image_id': image_id,
                    'interactions': list(chain.from_iterable(interactions))
                }
                    for image_id, interactions in collected.items()]
        return result_data

    def __make_unique_vocab(self, values):
        singular_words = [itm.split(' ') for itm in values]
        unique_words = list(set(chain.from_iterable(singular_words)))
        in_between = list(filterfalse(lambda i: not i, unique_words))
        return sorted(in_between)

    def __make_vocab(self):
        collection = self.flatten.word_collection
        in_between = {
            db_name: self.__make_unique_vocab(value)
            for db_name, value in collection.items()
        }
        return in_between

    def __call__(self, data: dict):
        restructured = self.__restructure(data)
        return {
            key: list(sorted(
                self.__normalize_length(value),
                key=lambda x: x['image_id']))
            for key, value in restructured.items()
        }, self.__make_vocab()
