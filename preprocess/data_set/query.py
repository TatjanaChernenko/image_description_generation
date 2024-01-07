
from typing import Dict, List
from collections import defaultdict
from itertools import chain

from .coco import Coco, CocoA, DataSetBase

DATA_SETS_MAP: Dict[str, DataSetBase] = {
    'coco-a': CocoA,
    'coco': Coco,
}


class Query():
    data_sets: Dict[str, List[DataSetBase]] = defaultdict(list)

    def __initialize(self, data_sets: List[str]):
        for data_set in data_sets:
            if data_set in self.data_sets.keys():
                continue
            self.data_sets[data_set].append(DATA_SETS_MAP[data_set]())
            if data_set == 'coco':
                self.data_sets[data_set]\
                    .append(DATA_SETS_MAP[data_set](train=False))

    def __intersect_image_ids(self, data_sets_names: List[str] = []):
        if len(data_sets_names) == 0:
            raise ValueError('Data Sets can not be empty')
        image_ids_data_sets = [
            list(chain.from_iterable(
                [d.get_all_image_ids() for d in self.data_sets[name]]
            ))
            for name in data_sets_names
        ]

        result = image_ids_data_sets[0]
        for idx, image_ids in enumerate(image_ids_data_sets):
            if idx == 0:
                continue
            result = [id for id in result if id in image_ids]

        return list(set(result))

    def __get_data(self,
                   preperation,
                   img_ids: List[str],
                   as_list=True):
        data_sets = preperation['data_sets']
        if as_list:
            return [
                dset.get_all(img_ids, annotators=preperation['annotators'])
                for data_set in data_sets
                for dset in self.data_sets[data_set]
            ]
        return {
            data_set: dset.get_all(img_ids)
            for data_set in data_sets
            for dset in self.data_sets[data_set]
        }

    def __call__(self, preperation, as_list=True):
        data_sets = preperation['data_sets']
        self.__initialize(data_sets)

        intersected_image_ids = self.__intersect_image_ids(
            data_sets_names=data_sets)
        return self.__get_data(preperation=preperation,
                               img_ids=intersected_image_ids,
                               as_list=as_list)
