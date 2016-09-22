import logging
from logging.handlers import TimedRotatingFileHandler,RotatingFileHandler
import os,sys

def init_log(debug):
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - <%(filename)s-%(funcName)s:%(lineno)d> : %(message)s')
  if debug:
      level=logging.DEBUG
  else:
      level=logging.INFO
  logbase = os.getcwd()+"/log/"
  os.system("mkdir -p %s" % logbase)
  log_file = logbase+"deepcake.log"
  logger = logging.getLogger("nomral")
  logger.setLevel(level)
  file_handler = TimedRotatingFileHandler(log_file,"midnight", 1, 30, None, True)
  console_handler = logging.StreamHandler()
  file_handler.setFormatter(formatter)
  console_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)
  logger.info("init logger success {}".format(4))
  return logger
