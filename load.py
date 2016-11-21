import struct
import scipy.misc
import numpy as np
import glob
from collections import defaultdict
import os
from os.path import join
from six.moves import cPickle as pickle

import local

size = 224 # must be a multiple of 32 to work with maxpooling in vgg16
num_classes = 60

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
			image = scipy.misc.imresize(image, (size, size))

			writer  = bmps_filepath.split("/")[-1].split(".")[0]
			write_image(tag_name, image, writer)

def get_classes():
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
	return classes


def bmps_to_pickle():
	classes = get_classes()

	for name in [ f for f in os.listdir(local.COMPETITION_GNT_PATH) if (f.startswith("C") and f.endswith("f-f")) ]:
		filepath = join(local.COMPETITION_GNT_PATH, name)
		writer  = filepath.split("/")[-1].split(".")[0]
	output = {
		"train_data": None,
		"train_classes": None,
		"valid_data": None,
		"valid_classes": None,
		"test_data": None,
		"test_classes": None,
	}

def main():
	# Produce bmps
	gnt_names = [ name for name in os.listdir(local.COMPETITION_GNT_PATH) if name.endswith(".gnt") ]
	for bmps_filepath in bmps_filepaths:
		write_gnt_to_bmps(bmps_filepath)

if __name__=="__main__":
	main()