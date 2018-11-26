import os, shutil

categ_cons = "kiss"

#categ_list = [categ_cons+"_unsafe", "normal"]
categ_list = [categ_cons+"_unsafe", categ_cons+"_safe"]

for categ in categ_list:

	#constants
	train_dir = "../data/training_data_"+categ_cons+"/train/"+ categ +"/"
	valid_dir = "../data/training_data_"+categ_cons+"/valid/"+ categ +"/"
	all_images_dir = "../data/input_"+categ_cons+"/"+ categ +"/"
#	all_images_dir = "../data/input_kiss"+"/"+ categ +"/"
	print(all_images_dir)

	#remove previous training data
	os.system('rm -rf  ' + train_dir)
	os.system('rm -rf  ' + valid_dir)

	#create appropriate folders
	os.makedirs(train_dir, exist_ok=True)
	os.makedirs(valid_dir, exist_ok=True)


	#walk all images and copy to appropriate folders
	i = 0
	for root, dirs, files in os.walk(all_images_dir):
		for file in files:
			if i%5 == 0:
				print(all_images_dir+str(file), valid_dir+file)
				shutil.copy(all_images_dir+str(file), valid_dir+file)
				i += 1
			else:
				print(all_images_dir+str(file), train_dir+file)
				shutil.copy(all_images_dir+str(file), train_dir+file)
				i += 1


# exit()
types = ["valid", "train"]
for type in types:

	#constants
	all_data_dir = "../data/training_data_"+categ_cons+"/"+type+"/"


	#walk all data
	all_categ_folders_name = []
	for root, dirs, files in os.walk(all_data_dir):
		i = 1
		for name in files:
			src = os.path.join(root, name)
			categ_name = root.split("/")[-1]
			all_categ_folders_name.append(categ_name)
			dest = all_data_dir+categ_name+str(i)+".jpg"
			# new_name = "../female/female" + str(i) + ".jpg"
			# dest = os.path.join(root, dest)
			shutil.move(src, dest)
			i += 1
			print(src)
			print(dest)

	#delete
	# print(set(all_categ_folders_name))
	for name in set(all_categ_folders_name):
		os.system("rm -rf "+all_data_dir+name)