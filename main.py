import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import Utils.Logger as logger
from Utils.Logger import log
from DataHandler import DataHandler
import tensorflow as tf
import random
from S2HyRec import S2HyRec

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')
	np.random.seed(215)
	random.seed(215)
	tf.set_random_seed(215)
	with tf.Session(config=config) as sess:
		s2hy = S2HyRec(sess, handler)
		s2hy.run()
