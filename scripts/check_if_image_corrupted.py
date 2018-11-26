
import os
from PIL import Image
import shutil
import cv2

list = []
ok = 0
corrupt = 0
k=0
'''root, dirs, files = os.walk("../../downloads/new_data_kiss_test/")
total_data=len(files)
'''

for root, dirs, files in os.walk("../data/training_data_kiss/train/"):
	print("k")
	for file in files:
		#print(root+"/"+file)
		k=k+1
		print(k)
		try:
			# img = Image.open(root+"/"+file)
			img = cv2.imread(root+"/"+file, cv2.IMREAD_COLOR)

			img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])

			 
			
			# if file == "kiss_safe100.jpg":
			# 	print(file)
			# im.verify()
			ok += 1
			# print("ok")
		except:
			# if file == "kiss_safe100.jpg":
			# 	print(file)
			
			corrupt += 1
			os.makedirs("../data/corrupt_images/", exist_ok=True)
			shutil.move(root+"/"+file, "../data/corrupt_images/"+file)
			print("corrupted: "+str(corrupt))
		# list.append(root+file)
		os.system('clear')


print(ok)
print(corrupt)