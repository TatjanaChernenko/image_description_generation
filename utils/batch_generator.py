from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ['Batch', 'iterator', 'infer_iterator']


class Batch(
    collections.namedtuple('Batch', ('initializer', 'source', 'target_in', 'target_out', 'source_sequence_len','target_sequence_len'))):
    pass


def infer_iterator(src_data, src_vocab_table, batch_size, eos, src_max_len=None):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_data = src_data.map(lambda src: tf.string_split([src]).values)

    if src_max_len:
        src_data = src_data.map(lambda src: src[:src_max_len])

    src_data = src_data.map(lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    src_data = src_data.map(lambda src: (src, tf.size(src)))

    def batching_fn(x):  # first size is batch size,second ist src, last is src_len
        return x.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
                              padding_values=(src_eos_id, 0))

    batched_data = batching_fn(src_data)
    batched_iter = batched_data.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()
    return Batch(initializer=batched_iter.initializer, source=src_ids, target_in=None, target_out=None,
                 source_sequence_len=src_seq_len, target_sequence_len=None)


def iterator(src_data, tgt_data, src_vocab_table, tgt_vocab_table, batch_size, sos, eos, random_seed, num_buckets,
             src_max_len=None, tgt_max_len=None, num_paralle_calls=4,
             output_buffer_size=None, skip_count=None, num_shards=1, shard_index=0, reshuffle_each_iteration=True):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_data = tf.data.Dataset.zip((src_data, tgt_data))

    src_tgt_data = src_tgt_data.shard(num_shards, shard_index)
    if skip_count is not None:
        src_tgt_data = src_tgt_data.skip(skip_count)

    src_tgt_data = src_tgt_data.shuffle(output_buffer_size, random_seed, reshuffle_each_iteration)

    src_tgt_data = src_tgt_data.map(
        lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_paralle_calls).prefetch(output_buffer_size)

    src_tgt_data = src_tgt_data.filter(lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_data = src_tgt_data.map(lambda src, tgt: (src[:src_max_len], tgt), num_paralle_calls).prefetch(
            output_buffer_size)

    if tgt_max_len:
        src_tgt_data = src_tgt_data.map(lambda src, tgt: (src, tgt[:tgt_max_len]),
                                        num_parallel_calls=num_paralle_calls).prefetch(output_buffer_size)

    src_tgt_data = src_tgt_data.map(lambda src, tgt: (
    tf.cast(src_vocab_table.lookup(src), tf.int32), tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
                                    num_parallel_calls=num_paralle_calls).prefetch(output_buffer_size)
    src_tgt_data = src_tgt_data.map(
        lambda src, tgt: (src, tf.concat(([tgt_sos_id], tgt), 0), tf.concat((tgt, [tgt_eos_id]), 0)),
        num_parallel_calls=num_paralle_calls).prefetch(output_buffer_size)
    src_tgt_data = src_tgt_data.map(lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
                                    num_parallel_calls=num_paralle_calls).prefetch(output_buffer_size)

    def batching_fn(x):  # shape (batch, src, tgt_in, tgt_out,src_len,tgt_len
        return x.padded_batch(batch_size, padded_shapes=(
        tf.TensorShape([None]), tf.TensorShape([None]),tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([])),
                              padding_values=(src_eos_id, tgt_eos_id, tgt_eos_id, 0, 0))

    if num_buckets > 1:

        def key_fn(unused_1, unused_2, unused_3, src_len, tgt_len):
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets

            else:
                bucket_width = 10

            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)

            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_fn(unused_key, windowed_data):
            return batching_fn(windowed_data)

        batched_set = src_tgt_data.apply(
                tf.contrib.data.group_by_window(key_func=key_fn, reduce_func=reduce_fn, window_size=batch_size))


    else:
        batched_set = batching_fn(src_tgt_data)

    batched_iter = batched_set.make_initializable_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (batched_iter.get_next())

    return Batch(initializer=batched_iter.initializer, source=src_ids, target_in=tgt_input_ids,
                 target_out=tgt_output_ids, source_sequence_len=src_seq_len, target_sequence_len=tgt_seq_len)
