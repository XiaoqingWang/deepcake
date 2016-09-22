import tensorflow as tf
import os

current_path = os.getcwd()
tfrecords_file_name = "pb/xaa.test.csv.tfrecords"
input_file = os.path.join(current_path, tfrecords_file_name)
print input_file

# Constrain the data to print
max_print_number = 100
print_number = 1

for serialized_example in tf.python_io.tf_record_iterator(input_file):
  example = tf.train.Example()
  example.ParseFromString(serialized_example)

  label = example.features.feature["label"].float_list.value
  categorical_features = example.features.feature["categorical_features"].bytes_list.value
  continuous_features = example.features.feature["continuous_features"].float_list.value
  print("Number: {}, label: {}\ncate-features: {}\nconti-features: {}".format(print_number, label,categorical_features,continuous_features))
  print("============================")

  # Return when reaching max print number
  if print_number > max_print_number:
    exit()
  else:
    print_number += 1
