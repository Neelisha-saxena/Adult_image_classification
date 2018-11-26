import os, shutil

categ_list = ["kiss_unsafe", "kiss_doubtfull", "normal"]

for categ in categ_list:

	#constants
	train_dir = "../data/training_data_kiss/train/"+ categ +"/"
	valid_dir = "../data/training_data_kiss/valid/"+ categ +"/"
	all_images_dir = "../data/input_kiss/"+ categ +"/"

	#create appropriate folders
	os.makedirs(train_dir, exist_ok=True)
	os.makedirs(valid_dir, exist_ok=True)


	#walk all images and copy to appropriate folders
	i = 0
	for root, dirs, files in os.walk(all_images_dir):
		for file in files:
			if i%4 == 0:
				print(all_images_dir+str(file), valid_dir+file)
				shutil.copy(all_images_dir+str(file), valid_dir+file)
				i += 1
			else:
				print(all_images_dir+str(file), train_dir+file)
				shutil.copy(all_images_dir+str(file), train_dir+file)
				i += 1