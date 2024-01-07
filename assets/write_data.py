from persistance.binary import Binary
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


def make_capt_vocab(lines):
    vocab = set()
    for capt in lines:
        for w in tokenizer.tokenize(capt.lower()):
                vocab.add(w)

    return vocab

def make_attr_vocab(lines):
    vocab = set()
    for line in lines:
        if type(line) == list:
            for w in line:
                for w_ in w.split():
                    vocab.add(w_)
        elif type(line) == str:
            for w_ in line.split():
                vocab.add(w_)

    return vocab

def write_coco(data_path,out_dir):

    end_train = int(len(data_dict)*0.7)
    end_dev = int(len(data_dict)*0.9)


    vocab_attr = set()
    vocab_capt = set()

    capt_train = open(out_dir + 'train.capt', 'w')
    attr_train = open(out_dir + 'train.attr', 'w')
    attr_dev = open(out_dir + 'dev.attr', 'w')
    capt_dev = open(out_dir + 'dev.capt', 'w')
    attr_test = open(out_dir + 'test.attr', 'w')
    capt_test = open(out_dir + 'test.capt', 'w')

    image_id = open(out_dir + 'image_id', 'w')
    
    data_dict = Binary.load(data_bin)
    for image in data_dict['coco'][0:end_train]:
        for capt in image['captions']:
            if capt != '--':
                image_id.write('train' + ' ' + str(image['image_id']) + '\n')
                attr = [cat for cat in image['categories'] if cat != '--']
                attr_train.write(' '.join(attr) + '\n')
                capt_train.write(' '.join(tokenizer.tokenize(capt.replace('\n', ' ').lower())) + '\n')
            for w in tokenizer.tokenize(capt.lower()):
                vocab_capt.add(w)
        for w in image['categories']:
            for w_ in w.split():
                vocab_attr.add(w_)

    for image in data_dict['coco'][end_train:end_dev]:
        for capt in image['captions']:
            if capt != '--':
                    image_id.write('dev' + ' ' + str(image['image_id']) + '\n')
                    attr = [cat for cat in image['categories'] if cat != '--']
                    attr_dev.write(' '.join(attr) + '\n')
                    capt_dev.write(' '.join(tokenizer.tokenize(capt.replace('\n', ' ').lower())) + '\n')
            for w in tokenizer.tokenize(capt.lower()):
                vocab_capt.add(w)
        for w in image['categories']:
            for w_ in w.split():
                vocab_attr.add(w_)

    for image in data_dict['coco'][end_dev:len(data_dict)]:
        for capt in image['captions']:
            if capt != '--':
                image_id.write('test' + ' ' + str(image['image_id']) + '\n')
                attr = [cat for cat in image['categories'] if cat != '--']
                attr_test.write(' '.join(attr) + '\n')
                capt_test.write(' '.join(tokenizer.tokenize(capt.replace('\n', ' ').lower())) + '\n')
            for w in tokenizer.tokenize(capt.lower()):
                    vocab_capt.add(w)
        for w in image['categories']:
                for w_ in w.split():
                    vocab_attr.add(w_)

    attr_train.close()
    capt_train.close()
    attr_dev.close()
    capt_dev.close()
    attr_test.close()
    capt_test.close()
    image_id.close()

    with open(out_dir + 'vocab.attr', 'w') as f:

        for w in vocab_attr:
            f.write(w + '\n')

    with open(out_dir + 'vocab.capt', 'w') as f:

        for w in vocab_capt:
            f.write(w + '\n')


def write_coco_a(data_path,out_dir,image_id_file, num_interaction=None):

    attr_train = open(out_dir + 'train.attr', 'w')

    attr_dev = open(out_dir+ 'dev.attr', 'w')

    attr_test = open(out_dir + 'test.attr', 'w')

    coco_a_dict = {}  # without '--'
    vocab_attr = set()
    data_dict = Binary.load(data_path)
    for image in data_dict['coco-a']:
        if image['image_id'] not in coco_a_dict:
            clear = list()
            for inter in image['interactions']:
                clear.append([attr for attr in inter if attr != '--'])
            for inter in clear:
                for a in inter:
                    for w in a.split():
                        vocab_attr.add(w)

            coco_a_dict[image['image_id']] = clear

            if num_interaction:
                assert type(num_interaction) == int
                coco_a_dict[image['image_id']] = clear[:num_interaction]

    with open(image_id_file) as f:
        for line in f.readlines():
            new_l = line.split()
            pre = new_l[0]
            image = int(new_l[1])
            if pre == 'train':
                attr = ' </s> '.join([' '.join(one) for one in coco_a_dict[image] if one != []])
                attr_train.write(attr + '\n')
            elif pre == 'dev':
                attr = ' </s> '.join([' '.join(one) for one in coco_a_dict[image] if one != []])
                attr_dev.write(attr + '\n')
            elif pre == 'test':
                attr = ' </s> '.join([' '.join(one) for one in coco_a_dict[image] if one != []])
                attr_test.write(attr + '\n')

    attr_train.close()
    attr_dev.close()
    attr_test.close()

    with open(out_dir + 'vocab.attr', 'w') as f:
        for a in vocab_attr:
            f.write(a + '\n')









