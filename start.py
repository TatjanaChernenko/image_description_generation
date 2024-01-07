#!/usr/bin/env python3

import sys

import tensorflow as tf

from persistance import Binary
from preprocess import Prepare
from preprocess.data_set import Coco, CocoA, Download
from utils import build_argument_parser
import attr2capt


def main(args, unparsed):
    if args.preprocess:
        Download.all()
        prepare = Prepare()
        prepare.run()

    print(args)
    attr2capt.run(id=args.id,
                  name=args.name,
                  training=args.training,
                  inference=args.inference,
                  decode_attr=args.decode_attr)


if __name__ == '__main__':
    arguments, unparsed = build_argument_parser()
    main(arguments, unparsed)
