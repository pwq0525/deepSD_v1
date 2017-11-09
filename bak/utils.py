"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

from read_nc import get_data
import tensorflow as tf

def get_names(config):
  return os.listdir(config.input)

def preprocess(config):

  input_ = get_data(config.input)
  feature_ = get_data(config.feature)
  dex = np.isnan(feature_)
  feature_[dex] = 0
  feature_ = feature_/3000.
  label_ = get_data(config.label)
  dex = np.isnan(label_)
  label_[dex] = 0
  dex = np.isnan(input_)
  input_[dex] = 0
  #img_ = get_data(config.input)
  scale0 = config.input_size/input_.shape[-1]
  img_ = scipy.ndimage.zoom(input_,(1,scale0,scale0))
  #img_ = np.maximum(img_,0)
  
  #dex = np.isnan(img_)
  #img_[dex] = 0
  #print("input:",input_.shape)
  #print("feature:",feature_.shape)
  #print("label:",label_.shape)
  
  # resize the input data and feature data from input_size to image_size
  scale = config.image_size/config.input_size
  img_ = scipy.ndimage.interpolation.zoom(img_, (1,scale,scale))
  img_ = np.maximum(img_,0)
  
  scale1 = config.image_size/feature_.shape[-1]
  feature_ = scipy.ndimage.zoom(feature_,(1,scale1,scale1))

  scale2 = config.label_size/label_.shape[-1]
  label_ = scipy.ndimage.zoom(label_, (1,scale2,scale2))
  label_ = np.maximum(label_,0)
  label_=label_.reshape(-1,config.label_size,config.label_size,1)

  shap = img_.shape
  image_ = np.zeros((shap[0],shap[1],shap[2],2))
  image_[:, :, :, 0] = img_
  image_[:, :, :, 1] = feature_
  label_ = label_
  return image_, label_

def input_setup(config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path

  '''
  if config.is_train:
    data = prepare_data(dataset="Train")
  else:
    data = prepare_data(dataset="Test")
  '''

  sub_input_sequence = []
  sub_label_sequence = []
  nxy = []
  #padding = abs(config.image_size - config.label_size) / 2 # 6
  #padding = int(padding)
  
  if config.is_train:

    input, label = preprocess(config)
    shape_i = input.shape
    shape_l = label.shape

    patch_size = config.patch_size
    patch_size_l = config.patch_size_l
    padding = (patch_size-patch_size_l)//2

    input_ = np.zeros((shape_i[0], shape_i[1]+padding+config.stride, shape_i[2]+padding+config.stride, shape_i[3]))
    label_ = np.zeros((shape_l[0], shape_l[1]+padding+config.stride, shape_l[2]+padding+config.stride, shape_l[3]))
    input_[:,padding:-config.stride,padding:-config.stride,:] = input
    label_[:,padding:-config.stride,padding:-config.stride,:] = label

    for i in range(len(input_)):
      if len(input_[i].shape) == 3:
        h, w, _ = input_[i].shape
      else:
        h, w = input_[i].shape

      for x in range(0, h-config.patch_size+1, config.stride):
        for y in range(0, w-config.patch_size+1, config.stride):
          sub_input = input_[i, x:x+config.patch_size, y:y+config.patch_size] # [12 x 12]
          sub_label = label_[i, x+padding:x+padding+patch_size_l, y+padding:y+padding+patch_size_l] # [6 x 6]

          # Make channel value
          #sub_input = sub_input.reshape([patch_size, patch_size, 2])  
          #sub_label = sub_label.reshape([patch_size_l, patch_size_l, 1])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  else:
    input, label = preprocess(config)
    shape_i = input.shape
    shape_l = label.shape
    
    patch_size = config.patch_size
    patch_size_l = config.patch_size_l
    padding = (patch_size-patch_size_l)//2

    input_ = np.zeros((shape_i[0], shape_i[1]+padding+config.stride, shape_i[2]+padding+config.stride, shape_i[3]))
    label_ = np.zeros((shape_l[0], shape_l[1]+padding+config.stride, shape_l[2]+padding+config.stride, shape_l[3]))
    input_[:,padding:-config.stride,padding:-config.stride,:] = input
    label_[:,padding:-config.stride,padding:-config.stride,:] = label 

    for i in range(len(input_)):

      if len(input_[i].shape) == 3:
        h, w, _ = input_[i].shape
      else:
        h, w = input_[i].shape

      # Numbers of sub-images in height and width of image are needed to compute merge operation.
      nx = ny = 0 
      for x in range(0, h-config.patch_size+1, config.stride):
        nx += 1; ny = 0
        for y in range(0, w-config.patch_size+1, config.stride):
          ny += 1
          sub_input = input_[i, x:x+patch_size, y:y+patch_size] # [33 x 33]
          sub_label = label_[i, x+padding:x+padding+patch_size_l, y+padding:y+padding+patch_size_l] # [21 x 21]
        
          #sub_input = sub_input.reshape([config.image_size, config.image_size, 2])  
          #sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)
      nxy.append([nx,ny])
  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

  if not config.is_train:
    return nxy,arrdata,arrlabel
  else:
    return arrdata,arrlabel
    
def imsave(image, path):
  #return scipy.misc.imsave(path, image)
  return plt.imsave(path,image)

def merge(images, size, config):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w] = image

  return img[:config.label_size, :config.label_size, :]

def merge_input(images,size,config):
  h, w = config.patch_size_l, config.patch_size_l
  padd = (images.shape[1]-h)/2
  padd = int(padd)
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w] = image[padd:-padd,padd:-padd]

  return img[:config.label_size, :config.label_size, :]

