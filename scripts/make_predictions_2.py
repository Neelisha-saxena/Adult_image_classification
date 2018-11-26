import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../model/deploy_yahoo_test.prototxt'
PRETRAINED = '../output/caffe_kiss_model_6_iter_4000.caffemodel'

# load the model
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('../data/lmdb_kiss/mean.binaryproto').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
print("successfully loaded classifier")

for i in os.walk("../data/testing_data_kiss/"):
	for file in files:
		if ".jpg" not in file:
			continue
		# test on a image
		IMAGE_FILE = root+file
		input_image = caffe.io.load_image(IMAGE_FILE)
		# predict takes any number of images,
		# and formats them for the Caffe net automatically
		pred = net.predict([input_image])
		print(pred)
