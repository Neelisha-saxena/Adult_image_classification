'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''
import time
import os
import glob
import random
import numpy as np
import shutil

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
print("start")
time.sleep(10)
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    # print(img)
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

categ_cons = "kiss"
if not (os.path.exists("../data/lmdb_"+categ_cons)):
    os.makedirs("../data/lmdb_"+categ_cons )

train_lmdb = '../data/lmdb_'+categ_cons+'/train_lmdb'
validation_lmdb = '../data/lmdb_'+categ_cons+'/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)


train_data = [img for img in glob.glob("../data/training_data_"+categ_cons+"/train/*jpg")]
test_data = [img for img in glob.glob("../data/training_data_"+categ_cons+"/valid/*jpg")]

#merge train and test data to automatically divide them later
train_data.extend(test_data)
# print(len(train_data))
# exit()

#Shuffle train_data
random.shuffle(train_data)

print('Creating train_lmdb')

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        try:
        
            if in_idx %  6 == 0:
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # print("train")
            print(img_path)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if categ_cons+'_unsafe' in img_path:
                label = 0
            elif categ_cons+'_safe' in img_path:
                label = 1
            else:
                pass
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
            # print('{:0>5d}'.format(in_idx) + ':' + img_path)
             
        except:
            os.remove(img_path)
in_db.close()


print('\nCreating validation_lmdb')

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        try:
            if in_idx % 6 != 0:
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # print("valid")
            print(img_path)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if categ_cons+'_unsafe' in img_path:
                label = 0
            elif categ_cons+'_safe' in img_path:
                label = 1
            else:
                pass
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
            # print('{:0>5d}'.format(in_idx) + ':' + img_path)
             
             
        except:
            os.remove(img_path)
in_db.close()

print('\nFinished processing all images')