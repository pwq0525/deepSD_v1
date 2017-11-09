from utils import ( 
  input_setup, 
  imsave,
  merge,
  merge_input,
  get_data,
  get_names
)

from read_nc import *
from write_nc import *
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import cm

import numpy as np
import tensorflow as tf

class SRCNN(object):

  def __init__(self, 
               sess, 
               input_size=141,
               image_size=281,
               label_size=281, 
               batch_size=64,
               patch_size=33,
               patch_size_l = 21,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.input_size = input_size
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.patch_size_l = patch_size_l
    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    
    with tf.name_scope('image_input'):
      self.images = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 2], name='images')

      padd = (self.patch_size-self.patch_size_l)//2
      input1 = self.images[:,padd:-padd,padd:-padd,0]
      input2 = self.images[:,padd:-padd,padd:-padd,1]

      input1 = tf.reshape(input1,[-1, self.patch_size_l, self.patch_size_l, 1])
      input2 = tf.reshape(input2,[-1, self.patch_size_l, self.patch_size_l, 1])
      tf.summary.image('input', input1, 1)
      tf.summary.image('feature', input2, 1)

      self.labels = tf.placeholder(tf.float32, [None, self.patch_size_l, self.patch_size_l, self.c_dim], name='labels')
      tf.summary.image('label',self.labels,1)
    
    with tf.name_scope('weights'):
      self.weights = {
        'w1': tf.Variable(tf.random_normal([9, 9, 2, 128], stddev=1e-3), name='w1'),
        'w2': tf.Variable(tf.random_normal([1, 1, 128, 64], stddev=1e-3), name='w2'),
        'w3': tf.Variable(tf.random_normal([5, 5, 64, 1], stddev=1e-3), name='w3')
      }
      tf.summary.histogram('weights/w1',self.weights['w1'])
      tf.summary.histogram('weights/w2',self.weights['w2'])
      tf.summary.histogram('weights/w3',self.weights['w3'])

    with tf.name_scope('biases'):
      self.biases = {
        'b1': tf.Variable(tf.zeros([128]), name='b1'),
        'b2': tf.Variable(tf.zeros([64]), name='b2'),
        'b3': tf.Variable(tf.zeros([1]), name='b3')
      }
      tf.summary.histogram('biases/b1',self.biases['b1'])
      tf.summary.histogram('biases/b2',self.biases['b2'])
      tf.summary.histogram('biases/b3',self.biases['b3'])

    #with tf.name_scope('image_out'):
    self.pred = self.model()
    tf.summary.image('image_output/output',self.pred,1)

    # Loss function (MSE)
    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
      tf.summary.scalar('loss',self.loss)

    self.saver = tf.train.Saver()

  def train(self, config):
    if config.is_train:
      train_data, train_label = input_setup(config)
    else:
      nxy, train_data, train_label = input_setup(config)
    print(train_data.shape)
    print(train_label.shape)
    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

    #tf.initialize_all_variables().run()
    merged = tf.summary.merge_all()
    if not os.path.exists(config.log_dir):
      os.makedirs(config.log_dir)
    writer = tf.summary.FileWriter(config.log_dir,self.sess.graph)
    tf.global_variables_initializer().run()
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")
      loss_last = 10
      batch_idxs = len(train_data) // config.batch_size
      for ys in range(11,0,-1):
        if (batch_idxs % ys == 0):
          print("Every epoch record", ys ,"losses in the loss graph")
          record_idxs = batch_idxs // ys
          break

      for ep in range(config.epoch):
        # Run by batch images
        loss_total = 0
        np.random.seed(ep)
        np.random.shuffle(train_data)
        np.random.seed(ep)
        np.random.shuffle(train_label)
        for idx in range(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          result,_, err = self.sess.run([merged, self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

          if counter % 400 == 0:
            print("Epoch:[%2d], step:[%2d], time:[%4.2f], loss: [%.16f]" \
              % ((ep+1), counter, time.time()-start_time, err))
          loss_total+=err
          if counter % record_idxs == 0:
            writer.add_summary(result, counter)
          if (counter % 200==0):
            #loss_last = err
            self.save(config.checkpoint_dir, counter)
        loss_total = loss_total/batch_idxs
        print("Epoch: [%2d], total_loss: [%.12f]"%((ep+1),loss_total))
        if (loss_last>loss_total):
          print("----------------- Better ----------------")
        else:
          print("+++++++++++++++++ Worse +++++++++++++++++")
        loss_last=loss_total
    else:
      print("Testing...")
      bg=0
      en=0
      names = get_name(config.label)
      dimen = get_dimen(config)
      writedata = []
      for i in range(len(nxy)):
        bg = en
        en += nxy[i][0]*nxy[i][1]
        #a = self.pred.eval({self.images: train_data[bg:en], self.labels: train_label[bg:en]})
        #a = np.array(a)
        #aa+=[j for j in a]
        #self.sess.run(self.out,feed_dict={self.out:a})
        #result,err = self.sess.run([self.pred,self.loss],feed_dict={self.images: train_data[bg:en], self.labels: train_label[bg:en]})
        rresul,result,img,label,err = self.sess.run([merged,self.pred,self.images,self.labels,self.loss],feed_dict={self.images: train_data[bg:en], self.labels: train_label[bg:en]})
        #mm,imgout = self.sess.run([merged,self.out],feed_dict={self.out: result})
        print("Test loss "+str(i)+": ",err)
        #print(self.pred)
        #rresul = self.sess.run(merged)
        writer.add_summary(rresul,i)

        re = merge(result, nxy[i], config)
        re = re.squeeze()
        re = np.maximum(re,0)

        writedata.append(re)
        if (i+1) % 24 == 0:
          outnc = [i for i in dimen]
          outnc.append(writedata)
          writepath = os.path.join(config.sample_dir, names[(i+1)//24-1])
          writepath = os.path.join(os.getcwd(), writepath)
          writenc(writepath,outnc)
          writedata = []
          img0,img1 = img[:,:,:,0],img[:,:,:,1]
          img0,img1 = img0.reshape(img.shape[0],img.shape[1],img.shape[2],1),img1.reshape(img.shape[0],img.shape[1],img.shape[2],1)
          imagein = merge_input(img0,nxy[i],config)
          imagein = imagein.squeeze()
          featurein = merge_input(img1,nxy[i],config)
          featurein = featurein.squeeze()
          label = merge(label,nxy[i],config)
          label = label.squeeze()
          '''
          figure = plt.figure('Test',figsize=(20,20))
          plt.subplot(234)
          plt.title('input')
          plt.imshow(imagein,cmap=cm.s3pcpn)
          plt.subplot(232)
          plt.title('feature')
          plt.imshow(featurein,cmap=plt.cm.gist_earth)
          plt.subplot(235)
          plt.title('output')
          plt.imshow(re,cmap=cm.s3pcpn)
          plt.subplot(236)
          plt.title('label')
          fig_lab = plt.imshow(label,cmap=cm.s3pcpn)
          leg = plt.subplot(233)
          bar_lab = plt.colorbar(fig_lab) #, orientation='horizontal')
          bar_lab.set_label('mm')
          leg.remove()
          plt.show()
          '''
	
        #image_path = os.path.join(os.getcwd(), config.sample_dir)
        #image_path = os.path.join(image_path, "test_image"+str(i)+".png")
        #imsave(re, image_path)


  def model(self):
    conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
    conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
