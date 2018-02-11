"""
Produces folders with bmps generated of the HWDB1.1 Chinese character dataset of gnt files
"""

import struct
import scipy.misc
import numpy as np
import glob
from collections import defaultdict
import os
from os.path import join
from six.moves import cPickle as pickle

import local
from hsk import vocab

IMG_SIZE = 224 # must be a multiple of 32 to work with maxpooling in vgg16

TRAIN_SET_SIZE = 0.5
VALID_SET_SIZE = 0.3
TEST_SET_SIZE = 0.2

def write_image(tag_name, image, writer):
	writer_path = join(local.COMPETITION_GNT_PATH, writer)
	if not os.path.exists(writer_path):
		os.mkdir(writer_path)
	output_path = join(writer_path, "%s.bmp" % tag_name)
	# write image to bmp
	if not os.path.exists(output_path):
		scipy.misc.imsave(output_path, image)
		print output_path

def write_gnt_to_bmps(bmps_filepath):
	with open(bmps_filepath, "rb") as f:
		while True:
			packed_length = f.read(4)
			if packed_length == '' or packed_length == ' ' or packed_length == b'':
				break
			length = struct.unpack("<I", packed_length)[0]
			tag_name = f.read(2)
			tag_name = tag_name.decode("gb2312")
			width = struct.unpack("<H", f.read(2))[0]
			height = struct.unpack("<H", f.read(2))[0]

			raw_bytes = f.read(height*width)
			bytez = struct.unpack("{}B".format(height*width), raw_bytes)

			image = np.array(bytez).reshape(height, width)
			image = scipy.misc.imresize(image, (IMG_SIZE, IMG_SIZE))

			writer  = bmps_filepath.split("/")[-1].split(".")[0]
			write_image(tag_name, image, writer)

def get_classes(hsk_levels=(1,2,3,4,5,6)):
	# Not all writers have written examples for every classes so
	# determine the classes which are present for all writers
	classes = None
	bmps_directories = [ f for f in os.listdir(local.COMPETITION_GNT_PATH) if (f.startswith("C") and f.endswith("f-f")) ]
	for name in bmps_directories:
		filepath = join(local.COMPETITION_GNT_PATH, name.strip(".gnt"))
		class_names = set([ sub_name.strip(".bmp") for sub_name in os.listdir(filepath) if sub_name.endswith(".bmp") ])
		if classes == None:
			classes = class_names
		else:
			classes &= class_names
			
	if hsk_levels:
		classes = [ cl for cl in classes if vocab.get(cl) in hsk_levels ]

	return sorted(list(classes))


def open_image_as_array(filepath):
	with open(filepath, "rb") as f:
		img = scipy.misc.imread(f, flatten=True)
		for i in range(len(img)):
			for j in range(len(img[0])):
				img[i][j] = (img[i][j]/255.0) - 0.5
	return img

def bmps_to_pickle():
	classes = get_classes(hsk_levels=(1,2,3))
	num_classes = 100
	number_of_authors = 60
	classes = classes[:num_classes]
	class_labels = {label: i for i, label in enumerate(classes)}
	
	train_size = int(num_classes*number_of_authors*TRAIN_SET_SIZE)
	valid_size = int(num_classes*number_of_authors*VALID_SET_SIZE)
	test_size = int(num_classes*number_of_authors*TEST_SET_SIZE)

	# import ipdb; ipdb.set_trace()
	train_data = np.ndarray((train_size, IMG_SIZE, IMG_SIZE), dtype=np.float32)
	valid_data = np.ndarray((valid_size, IMG_SIZE, IMG_SIZE), dtype=np.float32)
	test_data = np.ndarray((test_size, IMG_SIZE, IMG_SIZE), dtype=np.float32)
	train_labels = np.ndarray(train_size, dtype=np.int32)
	valid_labels = np.ndarray(valid_size, dtype=np.int32)
	test_labels = np.ndarray(test_size, dtype=np.int32)

	random_indexes = list(np.arange(train_size+valid_size+test_size))
	np.random.shuffle(random_indexes)

	train_i = valid_i = test_i = 0

	bmps_directories = sorted([ f for f in os.listdir(local.COMPETITION_GNT_PATH) if (f.startswith("C") and f.endswith("f-f")) ])
	for name in bmps_directories:
		print "author: %s" % name
		bmps_directory = join(local.COMPETITION_GNT_PATH, name)
		bmps_names = [ sub_name for sub_name in os.listdir(bmps_directory) if sub_name.endswith(".bmp") and sub_name.strip(".bmp") in classes ]
		assert len(bmps_names) == 100
		np.random.shuffle(bmps_names)
		for i, sub_name in enumerate(bmps_names):
			bmp_path = join(local.COMPETITION_GNT_PATH, name, sub_name)
			class_char = sub_name.strip(".bmp")
			img = open_image_as_array(bmp_path)
			if i < (TRAIN_SET_SIZE*num_classes):
				np.copyto(train_data[train_i], img)
				train_labels[train_i] = class_labels[class_char]
				train_i += 1
			elif (TRAIN_SET_SIZE*num_classes) <= i < ((TRAIN_SET_SIZE+VALID_SET_SIZE)*num_classes):
				np.copyto(valid_data[valid_i], img)
				valid_labels[valid_i] = class_labels[class_char]
				valid_i += 1
			else:
				np.copyto(test_data[test_i], img)
				test_labels[test_i] = class_labels[class_char]
				test_i += 1


	assert train_i == train_size
	assert valid_i == valid_size
	assert test_i == test_size
	
	output = {
		"train_data": train_data,
		"train_labels": train_labels,
		"valid_data": valid_data,
		"valid_labels": valid_labels,
		"test_data": test_data,
		"test_labels": test_labels,
	}

	# import ipdb; ipdb.set_trace()
	output_path = join(local.COMPETITION_GNT_PATH, "hsk_100_dataset.pickle")
	f = open(output_path, "wb")
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
	f.close()

def reformat(data, labels, num_channels=1):
	_, x_size, y_size = data.shape

	args = (-1, x_size, y_size, num_channels)
	data = data.reshape(args).astype(np.float32)

	num_labels = len(set(labels))
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return data, labels

def load_hsk_data():
	with open(local.HSK_DATA_PATH, "rb") as f:
		 data = pickle.load(f)
	return data

def write_all_gnts_to_bmps():
	gnt_names = [ name for name in os.listdir(local.COMPETITION_GNT_PATH) if name.endswith(".gnt") ]
	bmps_filepaths = [ join(local.COMPETITION_GNT_PATH, name) for name in gnt_names ]
	for bmp_path in bmps_filepaths:
		write_gnt_to_bmps(bmp_path)

def main():
	# write_all_gnts_to_bmps()
	bmps_to_pickle()

if __name__=="__main__":
	main()