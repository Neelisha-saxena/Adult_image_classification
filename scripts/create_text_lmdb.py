import os, random, glob

train_data = [img for img in glob.glob("../input/train/*jpg")]

random.shuffle(train_data)

for idx, img in enumerate(train_data):
	print(idx, img)
	if idx % 6 == 0:
		continue
	else:
		continue

#print(train_data)