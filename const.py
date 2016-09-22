import tensorflow as tf
import os,sys

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 0.015, "learning rate")
flags.DEFINE_integer("epoch_number", None, "epoch") ###### TODO(yuebin): bug here
flags.DEFINE_integer("batch_size", 1024, "batch size")
flags.DEFINE_integer("validate_batch_size", 1024, "")
flags.DEFINE_integer("thread_number", 16, "thread num to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "indicates min_after_dequeue of shuffle queue")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("model", "wide_n_deep", "wide, deep, wide_n_deep")
flags.DEFINE_string("optimizer", "adagrad", "optimizer to train")
flags.DEFINE_integer("steps_to_validate", 100,
                     "Steps to validate and print loss")
flags.DEFINE_string("mode", "train_from_scratch",
                    "train, train_from_scratch")

flags.DEFINE_string("train_pattern", "data/pb/x*.train.csv.tfrecords","")
flags.DEFINE_string("test_pattern", "data/pb/x*.test.csv.tfrecords","")

#distributed
tf.app.flags.DEFINE_string("ps_hosts", "","")
tf.app.flags.DEFINE_string("worker_hosts", "","")
tf.app.flags.DEFINE_string("job_name", "", "'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "")

CATEGORICAL_FEATURES_SIZE=5
CONTINUOUS_FEATURES_SIZE=7
FEATURE_SIZE = CATEGORICAL_FEATURES_SIZE + CONTINUOUS_FEATURES_SIZE

LABEL_SIZE = 2
BUCKET_SIZE = 2000000

train_pattern = FLAGS.train_pattern
test_pattern = FLAGS.test_pattern
learning_rate = FLAGS.learning_rate
epoch_number = FLAGS.epoch_number
thread_number = FLAGS.thread_number
batch_size = FLAGS.batch_size
validate_batch_size = FLAGS.validate_batch_size
min_after_dequeue = FLAGS.min_after_dequeue
capacity = thread_number * batch_size + min_after_dequeue
mode = FLAGS.mode
checkpoint_dir = FLAGS.checkpoint_dir
tensorboard_dir = FLAGS.tensorboard_dir
if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)
if not os.path.exists(tensorboard_dir):
  os.makedirs(tensorboard_dir)

input_units = FEATURE_SIZE
hidden1_units = 512
hidden2_units = 128
hidden3_units = 64
output_units = LABEL_SIZE

NUM_THREADS=32
