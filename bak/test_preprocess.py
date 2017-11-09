from utils import *
from read_nc import get_data

import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 20000, "Number of epoch [10]")

flags.DEFINE_string("input","input_141","input directory")
flags.DEFINE_string("label","label_141","label directory")
flags.DEFINE_string("feature","dem","feature directory")

flags.DEFINE_integer("input_size", 141, "The size of input image to use[141]")
flags.DEFINE_integer("image_size", 281, "The size of image to use [281]")
flags.DEFINE_integer("label_size", 281, "The size of label to produce [281]")
flags.DEFINE_integer("patch_size", 33, "the size of sub images")
flags.DEFINE_integer("patch_size_l", 21, "the size of sub label images")

flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("stride", 12, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "output_281", "Name of sample directory [output30]")
flags.DEFINE_string("log_dir","log","Name of tensorboard dir[log]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
	#pp.pprint(flags.FLAGS.__flags)
	image,label = preprocess(FLAGS)
	#image = get_data(FLAGS.input)
	#label = get_data(FLAGS.label)
	#feature = get_data(FLAGS.feature)
	#where_ara_nan = np.isnan(feature)
	#feature[where_ara_nan] = 0
	print(image.shape)
	print(label.shape)
	#print(feature.shape)

	minla = 100
	maxla = 0
	mindem = 100
	maxdem = 0
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			maxla = max(maxla, max(image[i,j,:,0]))
			minla = min(minla, min(image[i,j,:,0]))
			maxdem = max(maxdem, max(image[i,j,:,1]))
			mindem = min(mindem, min(image[i,j,:,1]))
			#print(image[i][j][-10:])
	print("image : ",maxla,minla,np.mean(image[:,:,:,0]))
	print("dem   : ",maxdem,mindem,np.mean(image[:,:,:,1]))

	minla = 1000
	maxla = 0
	for i in range(label.shape[0]):
		for j in range(label.shape[1]):
			maxla = max(maxla,max(label[i,j,:,0]))
			minla = min(minla,min(label[i,j,:,0]))
	print("label : ",maxla,minla,np.mean(label))
	
if __name__ == '__main__':
	tf.app.run()
