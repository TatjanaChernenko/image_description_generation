import os

from persistance import Binary
from utils import Configuration

from .data_set import Query
from .format import Format
from .text_format import TextFormat
from .sample import Sample


class Prepare():
    def __init__(self):
        self.config = Configuration()['prepare_data_sets']
        self.path = os.path.join(
            os.path.dirname(__file__), '..', 'assets', 'prepared_data')

        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.sample = Sample()
        self.textFormat = TextFormat()

    def make_vocab(self, data):
        pass

    def run(self):
        """
        Starts the preparation process for the data to MS training data
        """
        for preparation in self.config:
            if os.path.isfile(os.path.join(self.path,
                                           preparation['filename'])):
                continue

            query = Query()
            data = query(preperation=preparation,
                         as_list=False)

            format = Format(properties=preparation['properties'])
            done, _ = format(data)
            # split_data = self.sample(done)
            # # self.textFormat.save(split_data, vocab, done,
            #                      prefix=preparation['name'])

            Binary.save(done, os.path.join(self.path, preparation['filename']))
            # Binary.save(vocab,
            #             os.path.join(self.path, preparation['vocab_filename']))
