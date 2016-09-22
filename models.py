import tensorflow as tf
from config import *

def full_connect(inputs, weights_shape, biases_shape):
  with tf.device('/cpu:0'): # for better performance
    weights = tf.get_variable("weights",
                              weights_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases",
                             biases_shape,
                             initializer=tf.random_normal_initializer())
  return tf.matmul(inputs, weights) + biases

def do_relu(inputs, weights_shape, biases_shape):
  return tf.nn.relu(full_connect(inputs, weights_shape, biases_shape))

def deep_model(inputs):
  with tf.variable_scope("layer1"):
    layer = do_relu(inputs, [input_units, hidden1_units],
                              [hidden1_units])
  with tf.variable_scope("layer2"):
    layer = do_relu(layer, [hidden1_units, hidden2_units],
                              [hidden2_units])
  with tf.variable_scope("layer3"):
    layer = do_relu(layer, [hidden2_units, hidden3_units],
                              [hidden3_units])
  with tf.variable_scope("output"):
    layer = full_connect(layer, [hidden3_units, output_units], [output_units])
  return layer

def wide_model(inputs):
  with tf.variable_scope("logistic_regression"):
    layer = full_connect(inputs, [input_units, output_units], [output_units])
  return layer

def wide_and_deep_model(inputs):
  return wide_model(inputs) + deep_model(inputs)

