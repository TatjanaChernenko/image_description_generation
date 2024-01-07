"""TensorFlow model implementation."""
from __future__ import print_function

import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

import inference
import train
from utils import ModelSettings, evaluation_utils
from utils import general_utils as utils
from utils import vocab_utils

utils.check_tensorflow_version()

# reconstruct parameters from hparams, extend them
def extend_hparams(hparams):
    assert hparams.num_encoder_layers and hparams.num_decoder_layers
    if hparams.num_encoder_layers != hparams.num_decoder_layers:
        hparams.pass_hidden_state = False
        utils.print_out(f"""Num encoder layer {hparams.num_encoder_layers} is
                         different from num decoder layer
                         {hparams.num_decoder_layers}, so set
                         pass_hidden_state to False""")

    if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
        raise ValueError("For bi, num_encoder_layers %d should be even" %
                         hparams.num_encoder_layers)

    # we need residual layers if we initial lots of hidden layers
    num_encoder_residual_layers = 0
    num_decoder_residual_layers = 0
    if hparams.residual:
        if hparams.num_encoder_layers > 1:
            num_encoder_residual_layers = hparams.num_encoder_layers - 1
        if hparams.num_decoder_layers > 1:
            num_decoder_residual_layers = hparams.num_decoder_layers - 1

    hparams.add_hparam("num_encoder_residual_layers",
                       num_encoder_residual_layers)
    hparams.add_hparam("num_decoder_residual_layers",
                       num_decoder_residual_layers)

    if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
        raise ValueError("subword option must be either spm, or bpe")


    # make vocab table
    if hparams.vocab_prefix:
        src_vocab_file = hparams.vocab_prefix + "." + hparams.src
        tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
    else:
        raise ValueError("hparams.vocab_prefix must be provided.")

    src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
        src_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)

    if hparams.share_vocab:
        utils.print_out("  using source vocab for target")
        tgt_vocab_file = src_vocab_file
        tgt_vocab_size = src_vocab_size
    else:
        tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
            tgt_vocab_file,
            hparams.out_dir,
            check_special_token=hparams.check_special_token,
            sos=hparams.sos,
            eos=hparams.eos,
            unk=vocab_utils.UNK)
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
    hparams.add_hparam("src_vocab_file", src_vocab_file)
    hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

    hparams.add_hparam("src_embed_file", "")
    hparams.add_hparam("tgt_embed_file", "")
    if hparams.embed_prefix:
        src_embed_file = hparams.embed_prefix + "." + hparams.src
        tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

        if tf.gfile.Exists(src_embed_file):
            hparams.src_embed_file = src_embed_file

        if tf.gfile.Exists(tgt_embed_file):
            hparams.tgt_embed_file = tgt_embed_file

    # check out_dir
    if not tf.gfile.Exists(hparams.out_dir):
        tf.gfile.MakeDirs(hparams.out_dir)

    # evaluation
    for metric in hparams.metrics:
        hparams.add_hparam("best_" + metric, 0)  # larger is better
        best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
        hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

        if hparams.avg_ckpts:
            hparams.add_hparam("avg_best_" + metric, 0)  # larger is better
            best_metric_dir = os.path.join(
                hparams.out_dir, "avg_best_" + metric)
            hparams.add_hparam("avg_best_" + metric + "_dir", best_metric_dir)
            tf.gfile.MakeDirs(best_metric_dir)

    return hparams


def ensure_compatible_hparams(hparams, default_hparams, hparams_path):
    default_hparams = utils.maybe_parse_standard_hparams(
        default_hparams, hparams_path)

    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key, default_config[key])

    if default_hparams.override_loaded_hparams:
        for key in default_config:
            if getattr(hparams, key) != default_config[key]:
                utils.print_out("# Updating hparams.%s: %s -> %s" %
                                (key, str(getattr(hparams, key)),
                                 str(default_config[key])))
                setattr(hparams, key, default_config[key])
    return hparams

# inference input file
def start_inference(hparams, flags, out_dir, num_workers, jobid):
    hparams.inference_indices = None
    if flags['inference_list']:
        hparams.inference_indices = [
            int(token)
            for token in flags['inference_list']
        ]

    trans_file = flags.inference_output_file
    ckpt = flags.ckpt
    if not ckpt:
        ckpt = tf.train.latest_checkpoint(out_dir)
    inference.inference(ckpt, flags.inference_input_file,
                        trans_file, hparams, num_workers, jobid)

    # make evaluation if we have ref file
    ref_file = flags['inference_ref_file']
    if ref_file and tf.gfile.Exists(trans_file):
        for metric in hparams.metrics:
            score = evaluation_utils.evaluate(
                ref_file,
                trans_file,
                metric,
                hparams.subword_option)
            utils.print_out("  %s: %.1f" % (metric, score))

# inference one attributes list
def start_decode(name, input_attr):
    settings = ModelSettings()
    args = settings.get(name=name, id=id)
    hparams = tf.contrib.training.HParams(**args['hparameter'])  # noqa
    hparams = extend_hparams(hparams)
    
    attribute_list = list()
    attribute_list.append(input_attr)
    
    out_dir = hparams.out_dir

    # get the pretrained model
    ckpt = hparams.ckpt

    if not ckpt:
        ckpt = tf.train.latest_checkpoint(out_dir)

    decoded_s = inference.inference_message(ckpt, attribute_list, hparams)
    for out_s in decoded_s:
        print('New caption: ' + out_s, end="\n", file=sys.stdout)


def run(name=None, id=None, training=False, inference=False, decode_attr=False):

    settings = ModelSettings()
    args = settings.get(name=name, id=id)
    flags = args['flags']
    num_workers = flags['num_workers']
    random_seed = args['hparameter']['random_seed']
    jobid = flags['jobid']
    out_dir = args['hparameter']['out_dir']

    utils.print_out("# Job id %d" % jobid)
    if random_seed is not None and random_seed > 0:
        utils.print_out("# Set random seed to %d" % random_seed)
        random.seed(random_seed + jobid)
        np.random.seed(random_seed + jobid)

    if not tf.gfile.Exists(out_dir):
        tf.gfile.MakeDirs(out_dir)

    hparams = tf.contrib.training.HParams(**args['hparameter'])  # noqa
    hparams = extend_hparams(hparams)

    if inference:
        start_inference(hparams, flags, out_dir, num_workers, jobid)
    elif decode_attr:
        attr_input = input('Please give the attributes for this image: ')
        start_decode(hparams,attr_input)
    elif training:
        train.train(hparams, target_session="")
