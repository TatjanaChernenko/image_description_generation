import json
import os
from typing import Dict, List
from utils import Configuration
from itertools import chain

from pandas import DataFrame
from pandas.io.json import read_json
from pycocotools.coco import COCO

from utils import singleton


class DataSetBase():
    files: Dict[str, str] = dict()
    data: Dict[str, Dict] = dict()
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')

    def load(self):
        for key, file in self.files.items():
            with open(file) as fp:
                self.data[key] = json.load(fp)

    def get_all_image_ids(self):
        raise NotImplementedError('Please Implement if not done so already')

    def get_all(self, ids=[], path='image_id'):
        raise NotImplementedError('Please Implement if not done so already')


@singleton
class Coco(COCO, DataSetBase):
    def __init__(self, train=True):
        base = os.path.join(self.path, 'coco', 'annotations')
        super().__init__(os.path.join(base, 'instances_train2014.json')
                         if train else
                         os.path.join(base, 'instances_val2014.json'))
        self.coco_captions = COCO(os.path.join(base, 'captions_train2014.json')
                                  if train else
                                  os.path.join(base, 'captions_val2014.json'))

    def load(self):
        pass

    def get_all_image_ids(self):
        return self.getImgIds()

    def load_populated_anns(self, ids: List):
        anns = self.loadAnns(ids=ids)
        for ann in anns:
            ann['category'] = self.loadCats(ann['category_id'])[0]
        return anns

    def get_by_image_id(self, id: int, annotators=['1', '2', '3']):
        cap_anns_ids = self.coco_captions.getAnnIds(imgIds=[id])
        cap_anns = self.coco_captions.loadAnns(cap_anns_ids)
        anns_ids = self.getAnnIds(imgIds=[id])
        anns = self.load_populated_anns(anns_ids)
        img = self.loadImgs(ids=id)[0]
        return {
            'image': img,
            'captions': cap_anns,
            'anns': anns,
        }

    def get_all(self, ids=[], path='image_id', annotators=['1', '2', '3']):
        return [self.get_by_image_id(id) for id in ids]


@singleton
class CocoA(DataSetBase):
    def __init__(self, coco=None, train=False):
        base = os.path.join(self.path, 'coco-a')
        self.files = {
            'coco_a': os.path.join(base, 'cocoa_beta2015.json'),
            'visual_verbnet': os.path.join(base, 'visual_verbnet_beta2015.json'),  # noqa
        }
        self.load()
        tmp = [
            *self.data['coco_a']['annotations'][str(1)],
            *self.data['coco_a']['annotations'][str(2)],
            *self.data['coco_a']['annotations'][str(3)],
        ]

        abc = list(
            {i['id']: i for i in tmp}.values())

        self.col = {
            '1': self.data['coco_a']['annotations'][str(1)],
            '123': abc,
            '3': self.data['coco_a']['annotations'][str(3)]
        }

        self.coco = coco if coco else Coco()
        self.number_of_mturks = str(1)

    def get_all_image_ids(self, annotators=['1', '2', '3']):
        return list(set([itm['image_id']
                         for itm in self.col["".join(annotators)]]))

    def __filter_annotations(self,
                             path: str,
                             to_filter: List,
                             annotators=['1', '2', '3']):
        return [
            itm for itm in self.col["".join(annotators)]
            if itm[path] in to_filter
        ]

    def __get_visual_action(self, ids: List[int]):
        visual_actions = self.data['visual_verbnet']['visual_actions']
        if len(ids) != 0 and isinstance(ids[0], int):
            return [action for action in visual_actions if action['id'] in ids]
        else:
            return ids

    def __get_visual_adverbs(self, ids: List[int]):
        visual_adverbs = self.data['visual_verbnet']['visual_adverbs']
        if len(ids) != 0 and isinstance(ids[0], int):
            return [action for action in visual_adverbs if action['id'] in ids]
        else:
            return ids

    def populate(self, element):
        element['visual_actions'] = self.__get_visual_action(
            element['visual_actions'])
        element['visual_adverbs'] = self.__get_visual_adverbs(
            element['visual_adverbs'])

        if element['object_id'] != -1:
            object_ann = self.coco.load_populated_anns(
                ids=[element['object_id']])[0]
            element['object'] = object_ann['category']
        else:
            element['object'] = None

        if element['subject_id'] != -1:
            subject_ann = self.coco.load_populated_anns(
                ids=[element['subject_id']])[0]
            element['subject'] = subject_ann['category']
        else:
            element['subject'] = None

        return element

    def __get_by(self, path: str, to_filter: List, annotators=['1', '2', '3']):
        interactions = self.__filter_annotations(
            path, to_filter, annotators=annotators)
        return [self.populate(anno) for anno in interactions]

    def get_by_id(self, id: int):
        annotation = self.__get_by('id', [id])
        if len(annotation) == 0:
            return None
        return annotation[0]

    def get_by_image_id(self, id: int, annotators=['1', '2', '3']):
        interactions = self.__get_by('image_id', [id], annotators=annotators)
        if len(interactions) == 0:
            return None
        return interactions

    def get_all(self, ids=[], path='image_id', annotators=['1', '2', '3']):
        self.annotators = annotators
        if len(ids) == 0:
            return [self.populate(inter)
                    for inter in self.col["".join(annotators)]]
        else:
            return self.__get_by(path,
                                 to_filter=ids,
                                 annotators=['1', '2', '3'])
