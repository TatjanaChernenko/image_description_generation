import json
import os
from glob import glob
from zipfile import ZipFile

import requests
from requests.auth import HTTPBasicAuth
from utils import Configuration


class Download():

    @staticmethod
    def all():
        config = Configuration()
        for data_set in config['data_sets']:
            Download(data_set)

    def __init__(self, data_set):
        self.data_set = data_set
        self.folder_path = os.path.dirname(__file__)
        self.run()

    def run(self):
        for elem in self.data_set['annotations']:
            elem['target_path'] = os.path.join(
                self.folder_path, '..', '..', elem['target_path'])

            if not self.__should_initialize(elem['target_path']):
                continue

            self.__download(elem['url'], elem['target_path'],
                            elem['filename'])
            self.__unzip(elem['target_path'], elem['filename'])
            for json_file in self.__get_json_files(elem['target_path']):
                self.__beautify(json_file)

    def __download(self, url, target_path, filename):
        auth = self.__make_auth()
        final_file_path = os.path.join(target_path, filename)
        if os.path.isfile(final_file_path):
            os.unlink(final_file_path)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        response: requests.Response = requests.get(url, auth=auth, stream=True)

        with open(final_file_path, "wb") as output:
            print('Downloading: {} to {}'.format(filename, target_path))
            output.write(response.content)

    def __unzip(self, target_path, filename):
        print('Extracting {}'.format(filename))
        filepath = os.path.join(target_path, filename)
        if not os.path.isfile(filepath):
            raise Exception("File could not be found: {}".format(filepath))

        with ZipFile(filepath) as file:
            file.extractall(path=target_path)
            os.unlink(filepath)

    def __beautify(self, json_file):
        print('Beautifing {}'.format(json_file))
        with open(json_file, 'r') as fp:
            file = json.load(fp)
            os.unlink(json_file)

        with open(json_file, 'w') as fp:
            pretty = json.dumps(file, sort_keys=True, indent=4)
            fp.write(pretty)

    def __get_json_files(self, path):
        return glob(os.path.join(path, '**', '*.json'), recursive=True)

    def __should_initialize(self, path):
        folder = self.data_set['folder']
        validation = self.data_set['validation']
        target_path = os.path.join(path, folder) \
            if os.path.isdir(os.path.join(path, folder)) else path

        return not len(self.__get_json_files(target_path)) == validation

    def __make_auth(self):
        auth = self.data_set['auth']
        if not auth['password']:
            return None
        return HTTPBasicAuth(auth['username'], auth['password'])
