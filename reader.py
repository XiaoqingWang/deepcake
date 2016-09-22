import tensorflow as tf
from const import *

def read_and_decode_all_float(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          "label": tf.FixedLenFeature([], tf.float32),
          "features": tf.FixedLenFeature([FEATURE_SIZE], tf.float32),
      })
  label = features["label"]
  features = features["features"]
  return label, features


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          "label": tf.FixedLenFeature([], tf.float32),
          "categorical_features": tf.FixedLenFeature([CATEGORICAL_FEATURES_SIZE], tf.string),
          "continuous_features": tf.FixedLenFeature([CONTINUOUS_FEATURES_SIZE], tf.float32),
      })
  label = features["label"]
  continuous_features = features["continuous_features"]
  categorical_features = tf.cast(tf.string_to_hash_bucket(features["categorical_features"], BUCKET_SIZE), tf.float32)
  return label, tf.concat(0, [continuous_features, categorical_features])

filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("data/pb_join_zhubo2vec/x*.train.csv.tfrecords"),
    num_epochs=epoch_number)
label, features = read_and_decode(filename_queue)
batch_labels, batch_features = tf.train.shuffle_batch(
    [label, features],
    batch_size=batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

validate_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("data/pb_join_zhubo2vec/x*.test.csv.tfrecords"),
    num_epochs=epoch_number)

validate_label, validate_features = read_and_decode(validate_filename_queue)

validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
    [validate_label, validate_features],
    batch_size=validate_batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)
