import tensorflow as tf
import math
import os
import numpy as np
from config import *
#from reader import  read_and_decode


# Define parameters

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


# Read serialized examples from filename queue
def read_and_decode2(filename_queue):
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

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
	    filename_queue = tf.train.string_input_producer(
	        tf.train.match_filenames_once("data/pb/x*.train.csv.tfrecords"),
	        num_epochs=epoch_number)
	    label, features = read_and_decode(filename_queue)
	    batch_labels, batch_features = tf.train.shuffle_batch(
	        [label, features],
	        batch_size=batch_size,
	        num_threads=thread_number,
	        capacity=capacity,
	        min_after_dequeue=min_after_dequeue)
	    
	    validate_filename_queue = tf.train.string_input_producer(
	        tf.train.match_filenames_once("data/pb/x*.test.csv.tfrecords"),
	        num_epochs=epoch_number)
	    
	    validate_label, validate_features = read_and_decode(validate_filename_queue)
	    
	    validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
	        [validate_label, validate_features],
	        batch_size=validate_batch_size,
	        num_threads=thread_number,
	        capacity=capacity,
	        min_after_dequeue=min_after_dequeue)


            # Define the model
            input_units = FEATURE_SIZE
            hidden1_units = 10
            hidden2_units = 10
            output_units = 2

            # Hidden 1
            weights1 = tf.Variable(
                tf.truncated_normal([input_units, hidden1_units]),
                dtype=tf.float32,
                name='weights')
            biases1 = tf.Variable(
                tf.truncated_normal([hidden1_units]),
                name='biases',
                dtype=tf.float32)
            hidden1 = tf.nn.relu(tf.matmul(batch_features, weights1) + biases1)

            # Hidden 2
            weights2 = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units]),
                dtype=tf.float32,
                name='weights')
            biases2 = tf.Variable(
                tf.truncated_normal([hidden2_units]),
                name='biases',
                dtype=tf.float32)
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

            # Linear
            weights3 = tf.Variable(
                tf.truncated_normal([hidden2_units, output_units]),
                dtype=tf.float32,
                name='weights')
            biases3 = tf.Variable(
                tf.truncated_normal([output_units]),
                name='biases',
                dtype=tf.float32)
            logits = tf.matmul(hidden2, weights3) + biases3

            batch_labels = tf.to_int64(batch_labels)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, batch_labels)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            if FLAGS.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)

            # Compute accuracy
            accuracy_hidden1 = tf.nn.relu(tf.matmul(validate_batch_features,
                                                    weights1) + biases1)
            accuracy_hidden2 = tf.nn.relu(tf.matmul(accuracy_hidden1, weights2)
                                          + biases2)
            accuracy_logits = tf.matmul(accuracy_hidden2, weights3) + biases3
            validate_softmax = tf.nn.softmax(accuracy_logits)

            validate_batch_labels = tf.to_int64(validate_batch_labels)
            correct_prediction = tf.equal(
                tf.argmax(validate_softmax, 1), validate_batch_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Compute auc
            validate_batch_labels = tf.cast(validate_batch_labels, tf.int32)
            num_labels = 2
            sparse_labels = tf.reshape(validate_batch_labels, [-1, 1])
            derived_size = tf.shape(validate_batch_labels)[0]
            indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
            concated = tf.concat(1, [indices, sparse_labels])
            outshape = tf.pack([derived_size, num_labels])
            new_validate_batch_labels = tf.sparse_to_dense(concated, outshape,
                                                           1.0, 0.0)
            _, auc_op = tf.contrib.metrics.streaming_auc(
                validate_softmax, new_validate_batch_labels)

            # Define inference op
            inference_features = tf.placeholder("float", [None, FEATURE_SIZE])
            inference_hidden1 = tf.nn.relu(tf.matmul(inference_features,
                                                     weights1) + biases1)
            inference_hidden2 = tf.nn.relu(tf.matmul(inference_hidden1,
                                                     weights2) + biases2)
            inference_logits = tf.matmul(inference_hidden2, weights3) + biases3
            inference_softmax = tf.nn.softmax(inference_logits)
            inference_op = tf.argmax(inference_softmax, 1)

            saver = tf.train.Saver()
            steps_to_validate = FLAGS.steps_to_validate
            init_op = tf.initialize_all_variables()

            tf.scalar_summary('loss', loss)
            tf.scalar_summary('accuracy', accuracy)
            tf.scalar_summary('auc', auc_op)

            summary_op = tf.merge_all_summaries()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="./checkpoint/",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)

        with sv.managed_session(server.target) as sess:
            step = 0
            while not sv.should_stop() and step < 1000000:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                try:
                    while not coord.should_stop():
                        _, loss_value, step = sess.run([train_op, loss,
                                                        global_step])

                        if step % steps_to_validate == 0:
                            accuracy_value, auc_value, summary_value = sess.run(
                                [accuracy, auc_op, summary_op])
                            print(
                                "Step: {}, loss: {}, accuracy: {}, auc: {}".format(
                                    step, loss_value, accuracy_value,
                                    auc_value))

                except tf.errors.OutOfRangeError:
                    print("Done training after reading all data")
                finally:
                    coord.request_stop()

                coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
