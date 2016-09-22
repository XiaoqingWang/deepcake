import os,sys
import hashlib
import multiprocessing

def md5(key):
    m = hashlib.md5()
    m.update(key)
    return m.hexdigest()

current_path = os.getcwd()
print current_path

def process_one(input_file, output_file):
    test = None
    train = None
    print "input[%s] output[%s]" % (input_file,output_file)

    for line in open(input_file, "r"):
      item = line.replace("None","0").replace("u","").strip().split("\t")
      if item[0] == "day": #skip header
          continue
      day = item[0]

      if day[:6] == "201609":
          tag = "test"
          if not test:
            test = open(output_file+".test.csv","w")
      else:
          tag = "train"
          if not train:
            train = open(output_file+".train.csv","w")
      
      uid = item[1]
      rid = item[2]
      nr1 = item[3]
      nr2 = item[4]
      nr3 = item[5]
      tail = item[6:]
      loc = tail[0]
      sex = int(float(tail[1]))
      province = tail[2]
      rtype = int(float(tail[3]))
      score = int(float(tail[4]))
      angelcount = int(float(tail[5]))
      angellist = 0
      #angellist = tail[6]
      livetm = int(float(tail[7]))
      popular =int(float(tail[8]))
      rank = int(float(tail[9]))

      tail_str = "%s,"*10 % (loc,sex,province,rtype,score,angelcount,angellist,livetm,popular,rank)
      tail_str = tail_str.rstrip(",")
      if tag == "test":
          test.write("1,%s,%s,%s\n" % (uid,rid,tail_str))
          test.write("0,%s,%s,%s\n" % (uid,nr1,tail_str))
          test.write("0,%s,%s,%s\n" % (uid,nr2,tail_str))
          test.write("0,%s,%s,%s\n" % (uid,nr3,tail_str))
      elif tag =="train":
          train.write("1,%s,%s,%s\n" % (uid,rid,tail_str))
          train.write("0,%s,%s,%s\n" % (uid,nr1,tail_str))
          train.write("0,%s,%s,%s\n" % (uid,nr2,tail_str))
          train.write("0,%s,%s,%s\n" % (uid,nr3,tail_str))
    test.close()
    train.close()

if __name__ == "__main__":
  pool = multiprocessing.Pool(processes=32)
  workers= []

  for file in os.listdir(current_path+'/raw/'):
    current_path = os.getcwd()
    input_file = os.path.join(current_path+'/raw/', file)
    output_file = os.path.join(current_path+'/csv/', file)
    
    if file.startswith("x") and not file.endswith(".tfrecords"):
      workers.append(pool.apply_async(process_one, (input_file,output_file )))
  
  pool.close()
  pool.join()
