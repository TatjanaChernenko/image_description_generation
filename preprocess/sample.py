import math
from collections import defaultdict
from itertools import groupby, chain
from random import shuffle

import numpy as np


class Sample():
    def __init__(self, training_size=0.8, test_size=0.1, validation_size=0.1):
        assert training_size + test_size + validation_size == 1
        self.training_size = training_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.idxs = []

    def __separate(self, lst: list):
        length = len(lst)
        if len(self.idxs) == 0:
            self.idxs = list(range(length))
            shuffle(self.idxs)

        idxs = self.idxs

        upper_training = math.floor(length * self.training_size)
        upper_test = math.floor(
            length * self.training_size + length * self.test_size)
        training = list(np.take(lst, idxs[0:upper_training]))
        test = list(np.take(lst, idxs[upper_training + 1: upper_test]))
        validation = list(np.take(
            lst, idxs[upper_test + 1:]))

        return training, test, validation

    def __call__(self, data_sets):
        result = defaultdict(defaultdict)
        joined = list(chain.from_iterable(
            [data for data in data_sets.values()]))

        test = defaultdict(list)

        tmp = groupby(joined, key=lambda itm: itm['image_id'])

        # abc = [{
        #     'image_id': key,
        #     'data': [item for item in value]
        # } for key, value in tmp]

        for key, value in tmp:
            test[key].append([item for item in value])

        print(test[0:2])

        # for itm in tmp[0][1]:
        #     print(itm)

        # training, test, validation = self.__separate(data)
        # result[dset_name]['training'] = training
        # result[dset_name]['test'] = test
        # result[dset_name]['validation'] = validation
        return result
