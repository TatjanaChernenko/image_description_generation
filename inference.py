"""To perform inference on test set given a trained model."""
from __future__ import print_function

import codecs
import time

import tensorflow as tf

from seq2seqModel import attentionModel


from seq2seqModel import basicModel as seq_model
from seq2seqModel import modelBuilder
from utils import general_utils as utils
from utils import decode_seq

__all__ = ["load_data", "inference",
           "single_worker_inference", "multi_worker_inference"]


def _decode_inference_indices(model, sess, output_infer,
                              output_infer_summary_prefix,
                              inference_indices,
                              tgt_eos,
                              subword_option):
  utils.print_out("  decoding to output %s , num sents %d." %
                  (output_infer, len(inference_indices)))
  start_time = time.time()
  with codecs.getwriter("utf-8")(
      tf.gfile.GFile(output_infer, mode="wb")) as trans_f:
    trans_f.write("")  # write empty string to ensure file is created.
    for decode_id in inference_indices:
      m_outputs, infer_summary = model.decode(sess)

      # get text translation
      assert m_outputs.shape[0] == 1
      translation = decode_seq.decoded_sequence(
          m_outputs,
          sent_id=0,
          tgt_eos=tgt_eos,
          subword_option=subword_option)

      if infer_summary is not None:  # attention models
        image_file = output_infer_summary_prefix + str(decode_id) + ".png"
        utils.print_out("  save attention image to %s*" % image_file)
        image_summ = tf.Summary()
        image_summ.ParseFromString(infer_summary)
        with tf.gfile.GFile(image_file, mode="w") as img_f:
          img_f.write(image_summ.value[0].image.encoded_image_string)

      trans_f.write("%s\n" % translation)
      utils.print_out(translation + b"\n")
  utils.print_time("  done", start_time)


def load_data(inference_input_file, hparams=None):

  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  if hparams and hparams.inference_indices:
    inference_data = [inference_data[i] for i in hparams.inference_indices]

  return inference_data

# inference one attribute_list
def inference_message(ckpt,
                      attribute_list,
                      hparams,
                      scope=None):
    if hparams.attention_architecture == "standard":
        model_creator = attentionModel.encoder_decoder_attention
    else:
        raise ValueError("Unknown model architecture")
    infer_model = modelBuilder.create_infer_model(model_creator, hparams, scope)
    infer_data = attribute_list
    decoded_s = list()
    with tf.Session(
            graph=infer_model.graph, config=utils.get_config_proto()) as sess:
        loaded_infer_model = modelBuilder.load_model(
            infer_model.model, ckpt, sess, "infer")
        sess.run(
            infer_model.iterator.initializer,
            feed_dict={
                infer_model.src_placeholder: infer_data,
                infer_model.batch_size_placeholder: hparams.infer_batch_size
            })
        m_outputs, infer_summary = loaded_infer_model.decode(sess)
        if hparams.beam_width == 0:
            s = decode_seq.decoded_sequence(
                m_outputs,
                sent_id=0,
                tgt_eos=hparams.eos,
                subword_option=hparams.subword_option)
            # utils.print_bytes(s)

            if isinstance(s, bytes):
                s = s.decode("utf-8")
                out_s = s.encode("utf-8")
            if not isinstance(out_s, str):
                out_s = out_s.decode("utf-8")
                decoded_s.append(out_s)

        else:
            for n in range(m_outputs.shape[0]):
            #print(m_outputs[n])
                s = decode_seq.decoded_sequence(
                m_outputs[n],
                sent_id=0,
                tgt_eos=hparams.eos,
                subword_option=hparams.subword_option)
            # utils.print_bytes(s)

                if isinstance(s, bytes):
                    s = s.decode("utf-8")
                    out_s = s.encode("utf-8")
                if not isinstance(out_s, str):
                    out_s = out_s.decode("utf-8")
                    decoded_s.append(out_s)
    return decoded_s




# inference one input_file
def inference(ckpt,
              inference_input_file,
              inference_output_file,
              hparams,
              num_workers=1,
              jobid=0,
              scope=None):

  if hparams.inference_indices:
    assert num_workers == 1

  if not hparams.attention:
    model_creator = seq_model.encoder_decoder_basic
  elif hparams.attention_architecture == "standard":
    model_creator = attentionModel.encoder_decoder_attention

  else:
    raise ValueError("Unknown model architecture")
  infer_model = modelBuilder.create_infer_model(model_creator, hparams, scope)

  if num_workers == 1:
    single_worker_inference(
        infer_model,
        ckpt,
        inference_input_file,
        inference_output_file,
        hparams)
  else:
    multi_worker_inference(
        infer_model,
        ckpt,
        inference_input_file,
        inference_output_file,
        hparams,
        num_workers=num_workers,
        jobid=jobid)


def single_worker_inference(infer_model,
                            ckpt,
                            inference_input_file,
                            inference_output_file,
                            hparams):
  output_infer = inference_output_file

  # Read data
  infer_data = load_data(inference_input_file, hparams)

  with tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto()) as sess:
    loaded_infer_model = modelBuilder.load_model(
        infer_model.model, ckpt, sess, "infer")
    sess.run(
        infer_model.iterator.initializer,
        feed_dict={
            infer_model.src_placeholder: infer_data,
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        })
    if hparams.inference_indices:
      _decode_inference_indices(
          loaded_infer_model,
          sess,
          output_infer=output_infer,
          output_infer_summary_prefix=output_infer,
          inference_indices=hparams.inference_indices,
          tgt_eos=hparams.eos,
          subword_option=hparams.subword_option)
    else:
      decode_seq.decode_and_evaluate(
          "infer",
          loaded_infer_model,
          sess,
          output_infer,
          ref_file=None,
          metrics=hparams.metrics,
          subword_option=hparams.subword_option,
          beam_width=hparams.beam_width,
          tgt_eos=hparams.eos,
          num_translations_per_input=hparams.num_translations_per_input)


def multi_worker_inference(infer_model,
                           ckpt,
                           inference_input_file,
                           inference_output_file,
                           hparams,
                           num_workers,
                           jobid):
  assert num_workers > 1

  final_output_infer = inference_output_file
  output_infer = "%s_%d" % (inference_output_file, jobid)
  output_infer_done = "%s_done_%d" % (inference_output_file, jobid)

  infer_data = load_data(inference_input_file, hparams)

  total_load = len(infer_data)
  load_per_worker = int((total_load - 1) / num_workers) + 1
  start_position = jobid * load_per_worker
  end_position = min(start_position + load_per_worker, total_load)
  infer_data = infer_data[start_position:end_position]

  with tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto()) as sess:
    loaded_infer_model = modelBuilder.load_model(
        infer_model.model, ckpt, sess, "infer")
    sess.run(infer_model.iterator.initializer,
             {
                 infer_model.src_placeholder: infer_data,
                 infer_model.batch_size_placeholder: hparams.infer_batch_size
             })

    decode_seq.decode_and_evaluate(
        "infer",
        loaded_infer_model,
        sess,
        output_infer,
        ref_file=None,
        metrics=hparams.metrics,
        subword_option=hparams.subword_option,
        beam_width=hparams.beam_width,
        tgt_eos=hparams.eos,
        num_translations_per_input=hparams.num_translations_per_input)


    tf.gfile.Rename(output_infer, output_infer_done, overwrite=True)

    if jobid != 0: return

    with codecs.getwriter("utf-8")(
        tf.gfile.GFile(final_output_infer, mode="wb")) as final_f:
      for worker_id in range(num_workers):
        worker_infer_done = "%s_done_%d" % (inference_output_file, worker_id)
        while not tf.gfile.Exists(worker_infer_done):
          utils.print_out("  waitting job %d to complete." % worker_id)

        with codecs.getreader("utf-8")(
            tf.gfile.GFile(worker_infer_done, mode="rb")) as f:
          for translation in f:
            final_f.write("%s" % translation)

      for worker_id in range(num_workers):
        worker_infer_done = "%s_done_%d" % (inference_output_file, worker_id)
        tf.gfile.Remove(worker_infer_done)
