import shutil, os
categ_cons=" "

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