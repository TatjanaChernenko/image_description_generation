import os


class TextFormat():
    path = os.path.join(os.path.dirname(__file__),
                        '..', 'assets', 'leaning_input')
    prefix = 'none'

    def __init__(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def save_ids(self, data, name):
        with open(f"{self.path}/{self.prefix}-{name}.image_ids", 'a+') as file:
            file.write(data + '\n')

    def save_captions(self, data, type):
        ids = "\n".join([str(itm['image_id']) for itm in data])
        self.save_ids(ids, 'captions')

        with open(f"{self.path}/{self.prefix}-{type}.capt", 'w') as file:
            for elem in data:
                escaped = [item.replace("\n", "\\n")
                           for item in elem['captions']]
                captions = "\n".join(elem['captions'])
                file.write(captions + '\n')

    def save_interactions(self, data, type):
        ids = "\n".join([str(itm['image_id']) for itm in data])
        self.save_ids(ids, 'interactions')
        with open(f"{self.path}/{self.prefix}-{type}.attr", 'w') as file:
            for elem in data:
                interactions = " ".join([" </s> ".join(lst).replace("\n", "\\n")
                                         for lst in elem['interactions']])
                file.write(interactions + '\n')

    def save_vocab(self, data, type):
        data = [item.replace("\n", "\\n") for item in data]
        with open(f"{self.path}/{self.prefix}-{type}.vocab", 'w') as file:
            words = "\n".join(data)
            file.write(words)

    def save(self, split_data, vocab, data, prefix='none'):
        self.prefix = prefix
        nothing = [self.save_interactions(lst, key)
                   for key, lst in split_data['coco-a'].items()]

        nothing_2 = [self.save_captions(lst, key)
                     for key, lst in split_data['coco'].items()]

        nothing_3 = [self.save_vocab(lst, key) for key, lst in vocab.items()]
