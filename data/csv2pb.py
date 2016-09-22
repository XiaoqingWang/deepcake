import os,sys
import tensorflow as tf
import hashlib
import multiprocessing

def md5(key):
    m = hashlib.md5()
    m.update(key)
    return m.hexdigest()

def convert_tfrecords(input_filename, output_filename):
  current_path = os.getcwd()
  input_file = os.path.join(current_path+'/csv/', input_filename)
  output_file = os.path.join(current_path+'/pb/', output_filename)
  print("Start to convert {} to {}".format(input_file, output_file))

  writer = tf.python_io.TFRecordWriter(output_file)

  for line in open(input_file, "r"):
    item = line.split(",")
    label = float(item[0])

    #uid = item[1]
    #rid = item[2]
    #loc = item[3]
    #province = item[5]
    #angellist = item[9]

    #sex = float(item[4])
    #rtype = float(item[6])
    #score = float(item[7])
    #angelcount = float(item[8])
    #livetm = float(item[10])
    #popular = float(item[11])
    #rank = float(item[12])
    
    continus_index = [4,6,7,8,10,11,12]
    cata_index = [1,2,3,5,9]

    continus_features = [float(item[i]) for i in continus_index]
    cata_features = [item[i] for i in cata_index]

    example = tf.train.Example(features=tf.train.Features(feature={
        "label":
        tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        "categorical_features":
        tf.train.Feature(bytes_list=tf.train.BytesList(value=cata_features)),
        "continuous_features":
        tf.train.Feature(float_list=tf.train.FloatList(value=continus_features)),
    }))

    writer.write(example.SerializeToString())

  writer.close()
  print("Successfully convert {} to {}".format(input_file, output_file))


if __name__ == "__main__":
  pool = multiprocessing.Pool(processes=60)
  workers= []


  current_path = os.getcwd()
  for file in os.listdir(current_path+'/csv/'):
    if file.endswith(".csv") and not file.endswith(".tfrecords"):
      workers.append(pool.apply_async(convert_tfrecords, (file, file + ".tfrecords")))
  pool.close()
  pool.join()
