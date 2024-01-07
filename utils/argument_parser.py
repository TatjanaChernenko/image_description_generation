from argparse import ArgumentParser


def build_argument_parser():
    parser = ArgumentParser()
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--id', type=int,
                             help="select model by id (starting at 0)")
    model_group.add_argument('--name', type=str,
                             help="select model by name")

    runtype_group = parser.add_mutually_exclusive_group(required=True)
    runtype_group.add_argument('--train', help='do training',
                               dest='training', action='store_true')
    runtype_group.add_argument('--infer', help='do inference',
                               dest='inference', action='store_true')
    runtype_group.add_argument('--decode', help='decode attributes from one image',
                               dest='decode_attr', action='store_true')

    parser.add_argument('--no-preprocess', '-np', dest='preprocess',
                        action='store_false', help='disable dataset preprocessing')

    return parser.parse_known_args()
