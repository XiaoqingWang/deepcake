import datetime
import json
import os,sys
import numpy as np
import tensorflow as tf
from reader import *
from config import *
from logger import *
from models import *

logger = init_log(debug = True)

def model(inputs):
  if FLAGS.model == "wide":
    return wide_model(inputs)
  elif FLAGS.model == "deep":
    return deep_model(inputs)
  elif FLAGS.model == "wide_n_deep":
    return wide_and_deep_model(inputs)
  else:
    logger.error("unknown model")
    exit()

#loss here
logits = model(batch_features)
batch_labels = tf.to_int64(batch_labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, batch_labels)
loss = tf.reduce_mean(cross_entropy, name='loss')

logger.info("use the optimizer: %s" % FLAGS.optimizer)
if FLAGS.optimizer == "sgd":
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
elif FLAGS.optimizer == "adadelta":
  optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-08)
elif FLAGS.optimizer == "adagrad":
  optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1)
elif FLAGS.optimizer == "adam":
  optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
elif FLAGS.optimizer == "ftrl":
  optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=-0.5,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0)
elif FLAGS.optimizer == "rmsprop":
  optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10)
else:
  logger.error("unknow optimizer: %s" % FLAGS.optimizer)
  exit()

with tf.device("/cpu:0"):  # better than gpu
  global_step = tf.Variable(0, name='global_step', trainable=False)

train_op = optimizer.minimize(loss, global_step=global_step)

# metric here
tf.get_variable_scope().reuse_variables()
accuracy_logits = model(validate_batch_features)
validate_softmax = tf.nn.softmax(accuracy_logits)
validate_batch_labels = tf.to_int64(validate_batch_labels)
correct_prediction = tf.equal(tf.argmax(validate_softmax, 1), validate_batch_labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

validate_batch_labels = tf.cast(validate_batch_labels, tf.int32)
sparse_labels = tf.reshape(validate_batch_labels, [-1, 1])
derived_size = tf.shape(validate_batch_labels)[0]
indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
concated = tf.concat(1, [indices, sparse_labels])
outshape = tf.pack([derived_size, LABEL_SIZE])
new_validate_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
_, auc_op = tf.contrib.metrics.streaming_auc(validate_softmax,
                                             new_validate_batch_labels)

model_features = tf.placeholder("float", [None, FEATURE_SIZE])
model_logits = model(model_features)
model_softmax = tf.nn.softmax(model_logits)
model_op = tf.argmax(model_softmax, 1)

checkpoint_file = checkpoint_dir + "/checkpoint.ckpt"
steps_to_validate = FLAGS.steps_to_validate
init_op = tf.initialize_all_variables()
tf.scalar_summary('loss', loss)
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('auc', auc_op)

saver = tf.train.Saver()
keys_placeholder = tf.placeholder("float")
keys = tf.identity(keys_placeholder)
tf.add_to_collection("inputs", json.dumps({'key': keys_placeholder.name,
                                           'features':
                                           model_features.name}))
tf.add_to_collection("outputs", json.dumps({'key': keys.name,
                                            'softmax': model_softmax.name,
                                            'prediction': model_op.name}))

with tf.Session(
    config=tf.ConfigProto(
        inter_op_parallelism_threads=NUM_THREADS,
        intra_op_parallelism_threads=NUM_THREADS)) as sess:
  summary_op = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
  sess.run(init_op)
  sess.run(tf.initialize_local_variables())

  if mode == "train_continue" or mode == "train":
    if mode != "train":
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        logger.info("continue training from the model %s" % 
            ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    start_time = datetime.datetime.now()
    try:
      while not coord.should_stop():
        _, loss_value, step = sess.run([train_op, loss, global_step])
        if step % steps_to_validate == 0:
          accuracy_value, auc_value, summary_value = sess.run(
              [accuracy, auc_op, summary_op])
          end_time = datetime.datetime.now()
          logger.info("[%s] Step: %s, loss: %s, accuracy: %s, auc: %s" % (
              end_time - start_time, step, loss_value, accuracy_value,
              auc_value))

          writer.add_summary(summary_value, step)
          saver.save(sess, checkpoint_file, global_step=step)
          start_time = end_time
    except tf.errors.OutOfRangeError:
      logger.info("Done training after reading all data")
    finally:
      coord.request_stop()

    coord.join(threads)
