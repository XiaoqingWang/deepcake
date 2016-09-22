import tensorflow as tf
import os

current_path = os.getcwd()
tfrecords_file_name = "pb/xaa.test.csv.tfrecords"
input_file = os.path.join(current_path, tfrecords_file_name)
print input_file

max_ptr = 100
ptr = 1

for serialized_example in tf.python_io.tf_record_iterator(input_file):
  example = tf.train.Example()
  example.ParseFromString(serialized_example)

  label = example.features.feature["label"].float_list.value
  categorical_features = example.features.feature["categorical_features"].bytes_list.value
  continuous_features = example.features.feature["continuous_features"].float_list.value
  print("label: {}\ncate-features: {}\nconti-features: {}".format(label,categorical_features,continuous_features))
  print("============================")

  if ptr > max_ptr:
    exit()
  else:
    ptr += 1
