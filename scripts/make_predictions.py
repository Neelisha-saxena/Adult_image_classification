'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions_1.py
python_version  :2.7.11
'''

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

import shutil

caffe.set_mode_cpu() 

#Size of images
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

'''
Image processing helper function

'''

categ_cons='kiss'

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    try:
        #Histogram Equalization
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        #Image Resizing
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

        return img

    except:
        print("error")
        exit()


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('../data/lmdb_'+categ_cons+'/mean.binaryproto', "rb") as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/home/neelisha/Documents/vidooly/kiss/model/deploy_yahoo_test.prototxt',
                '/home/neelisha/Documents/vidooly/kiss/output/caffe_'+categ_cons+'_model_1/caffe_'+categ_cons+'_model_9_iter_9000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("/home/neelisha/Documents/vidooly/kiss/data/testing_data_kiss"+"/*jpg")]
print(len(test_img_paths))
np.random.shuffle(test_img_paths)

#Making predictions
test_ids = []
preds = []
for i, img_path in enumerate(test_img_paths):
    print(i)
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        print("***********")
        '''img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
                                
                                net.blobs['data'].data[...] = transformer.preprocess('data', img)
                                out = net.forward()
                                pred_probas = out['loss']
                                # print("foo_bar: "+str(pred_probas))
                        
                                test_ids = test_ids + [img_path.split('/')[-1][:-4]]
                                preds = preds + [pred_probas.argmax()]
                        
                                res_dir = "/home/neelisha/Documents/vidooly/kiss/data/testing_data_result"
                                os.makedirs(res_dir, exist_ok=True)
                        
                                os.makedirs(res_dir+"/"+categ_cons+"_unsafe" )
                                os.makedirs(res_dir+"/"+categ_cons+"_safe" )
                                #os.makedirs(res_dir+"/normal", exist_ok=True)
                                # os.makedirs(res_dir+"/none", exist_ok=True)
                        
                                if pred_probas[0][0] > pred_probas[0][1]:  #bikini
                                    shutil.copy(img_path, res_dir+"/"+categ_cons+"_unsafe/"+categ_cons[:4]+str(pred_probas[0][0])+",nrml"+str(pred_probas[0][1])+
                                        img_path.split('/')[-1])
                                    # print(img_path)
                                else:
                                    #pred_probas[0][2] > pred_probas[0][1] and pred_probas[0][2] > pred_probas[0][0]:  #kiss
                                    shutil.copy(img_path, res_dir+"/"+categ_cons+"_safe/"+categ_cons[:4]+str(pred_probas[0][0])+",nrml"+str(pred_probas[0][1])+
                                        img_path.split('/')[-1])
                                # elif pred_probas[0][0] > 0.7:  #normal
                                #     shutil.copy(img_path, res_dir+"/normal/"+img_path.split('/')[-1])
                                    # print("../data/result_testing/normal/"+img_path.split('/')[-1][:-4])
                                '''#else:
                                    #shutil.copy(img_path, res_dir+"/normal/"+img_path.split('/')[-1])  #normal'''
                        
                                #print(i)
                                #print(img_path)
                                #print(pred_probas.argmax())
                                #print('-------')
                        '''
    '''
    except:
        print("some error occured")

#Making submission file
'''
# with open("../files/testing_data_2.csv","w") as f:
#     f.write("id,label\n")
#     for i in range(len(test_ids)):
#         f.write(str(test_ids[i])+";"+str(preds[i])+"\n")
# f.close()
'''
