import sys

import tensorflow as tf

from persistance import Binary
from utils import build_argument_parser
import attr2capt

# another start file if we run on cluster server
def main(args, unparsed):
    attr2capt.run(id=args.id,
                  name=args.name,
                  training=args.training,
                  inference=args.inference,
                  decode_attr=args.decode_attr)


if __name__ == '__main__':
    arguments, unparsed = build_argument_parser()
    main(arguments, unparsed)
