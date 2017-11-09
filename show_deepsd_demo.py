from utils import *
from read_nc import get_data
from read_nc import get_name

import numpy as np
import tensorflow as tf
import scipy.ndimage
from mpl_toolkits.basemap import cm
import matplotlib.pyplot as plt
import pprint
import os
import matplotlib.gridspec as gridspec

flags = tf.app.flags

flags.DEFINE_string("result_2x","output_141","2x result directory")
flags.DEFINE_string("result_5x","output_281","5x result directory")
flags.DEFINE_string("label","valid_set","label directory")

flags.DEFINE_integer("input_size", 141, "The size of input image to use[141]")
flags.DEFINE_integer("size_2x", 281, "The size of image to use [281]")
flags.DEFINE_integer("size_5x", 701, "The size of label to produce [281]")
flags.DEFINE_integer("label_size", 701, "The size of label image")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
	pp.pprint(flags.FLAGS.__flags)
	img_2x = get_data(FLAGS.result_2x)
	img_5x = get_data(FLAGS.result_5x)
	label_img = get_data(FLAGS.label)

	scale0 = FLAGS.input_size/label_img.shape[-1]
	where_ara_nan = np.isnan(label_img)
	label_img[where_ara_nan] = 0
	input_img = scipy.ndimage.zoom(label_img,(1,scale0,scale0))
	input_img = np.maximum(input_img,0)
	#image = scipy.ndimage.zoom(label,(1,141/701,141/701))
	print(input_img.shape)
	print(img_2x.shape)
	print(img_5x.shape)
	names = get_name(FLAGS.label)
	for i in range(input_img.shape[0]):
		fig = plt.figure('DeepSD 5->1 km '+names[i//24]+'{0:4}/24'.format(i%24+1),figsize=(20,20))
		#gs = gridspec.GridSpec(1,32)
		plt.subplot(234)#(gs[0,0:9])#(234)
		plt.title('input')
		input_demo = plt.imshow(input_img[i],cmap=cm.s3pcpn)
		#plt.imshow(image[i],cmap=plt.cm.tab20)
		plt.subplot(235)#(gs[0,10:19])#(235)
		plt.title('2x output')
		plt.imshow(img_2x[i],cmap=cm.s3pcpn)
		#plt.imshow(feature[0],cmap=plt.cm.gist_earth)
		plt.subplot(236)#(gs[0,20:29])#(236)
		plt.title('5x output')
		plt.imshow(img_5x[i],cmap=cm.s3pcpn)
		#plt.imshow(label[i],cmap=plt.cm.tab20)
		leg = plt.subplot(233)#(gs[0,30:])#(233)
		cmbar = plt.colorbar(input_demo)
		cmbar.set_label('mm')
		leg.remove()
		plt.show()
		#plt.savefig('image/DeepSD_5-1_'+names[i//24].split('.')[0]+'_{0}|24'.format(i%24+1))
		#plt.close('all')

if __name__ == '__main__':
	tf.app.run()
