import tensorflow as tf

# make 4 attetion options, default is bahdanau
def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length, mode):

  if attention_option == "luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "scaled_luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        scale=True)
  elif attention_option == "bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "normed_bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        normalize=True)
  else:
    raise ValueError("Unknown attention option %s" % attention_option)

  return attention_mechanism

# create attention image and attention summary.
def _create_attention_images_summary(final_context_state):

  attention_images = (final_context_state.alignment_history.stack())

  attention_images = tf.expand_dims(
      tf.transpose(attention_images, [1, 2, 0]), -1) # Reshape to (batch, src_seq_len, tgt_seq_len,1)
  # Scale to range [0, 255]
  attention_images *= 255
  attention_summary = tf.summary.image("attention_images", attention_images)
  return attention_summary
