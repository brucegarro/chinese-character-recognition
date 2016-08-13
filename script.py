import struct
import scipy.misc
import numpy
import glob
from collections import defaultdict
import pdb
import os
from os.path import join

import local

side = 224 # must be a multiple of 32 to work with maxpooling in vgg16
full_data = defaultdict(list)
num_classes = 60
counter = 0
for filename in glob.glob("gnts/*.gnt"):
	print(filename)
	f = open(filename,"rb")
	while True:
		packed_length = f.read(4)
		if packed_length == '' or packed_length == ' ' or packed_length == b'':
			break
		print(packed_length)
		length = struct.unpack("<I", packed_length)[0]
		tag_name = f.read(2)
		# label = struct.unpack(">H", tag_name)[0]
		# label -= 0xb0a1 # CASIA labels start at 0xb0a1
		tag_name = tag_name.decode("gb2312")
		width = struct.unpack("<H", f.read(2))[0]
		height = struct.unpack("<H", f.read(2))[0]
		raw_bytes = f.read(height*width)
		bytes = struct.unpack("{}B".format(height*width), raw_bytes)
		existing_labels = full_data.keys()
		if (tag_name in existing_labels) or (len(existing_labels) < num_classes):
			image = numpy.array(bytes).reshape(height, width)
			image = scipy.misc.imresize(image, (side, side))
			# scipy.misc.imsave("%s.bmp" % (tag_name), image)
			image = (image.astype(float) / 256) - 0.5 # normalize to [-0.5,0.5] to avoid saturation
			full_data[tag_name].append(image)
			counter = counter + 1
	f.close()

def write_image(tag_name, image, gnt_filepath):
	# import ipdb; ipdb.set_trace()
	writer  = gnt_filepath.split("/")[-1].split(".")[0]
	writer_path = join(local.COMPETITION_GNT_PATH, writer)
	if not os.path.exists(writer_path):
		os.mkdir(writer_path)
	output_path = join(writer_path, "%s.bmp" % tag_name)
	# write image to bmp
	scipy.misc.imsave(output_path, image)
	print output_path

def gnt_to_bmp(filepath):
	counter = 0
	with open(filepath, "rb") as f:
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
			bytes = struct.unpack("{}B".format(height*width), raw_bytes)
			existing_labels = full_data.keys()
			if (tag_name in existing_labels) or (len(existing_labels) < num_classes):
				image = numpy.array(bytes).reshape(height, width)
				image = scipy.misc.imresize(image, (side, side))

				write_image(tag_name, image, filepath)
				# scipy.misc.imsave("%s.bmp" % (tag_name), image)
				image = (image.astype(float) / 256) - 0.5 # normalize to [-0.5,0.5] to avoid saturation
				full_data[tag_name].append(image)
				counter += 1

def main():
	for name in [ f for f in os.listdir(local.COMPETITION_GNT_PATH) if f.endswith(".gnt") ]:
		filepath = join(local.COMPETITION_GNT_PATH, name)
		print gnt_to_bmp(filepath)

if __name__=="__main__":
	main()

# structure of full_data -- dictionary with 60 keys,
# each of which is a character pointing to 60 examples
# each example is a 224x224 matrix
# Ignore the below

# dic = {}
# for line in open("gbdb").read().splitlines():
# 	if len(line) > 0:
# 		values = line.split("  ")
# 		dic[values[0]] = values[-1]

# file = open("C001-f-f.gnt", "rb")
# packed_length = file.read(4)
# length = struct.unpack("<I", packed_length)[0]
# label = file.read(2);

# width = struct.unpack("<H", file.read(2))[0]
# height = struct.unpack("<H", file.read(2))[0]
# print(length)
# print(label)
# print(width)
# print(height)
# print(bytearray.fromhex("a1b0").decode())

# file = open("C001-f-f.gnt", "rb")

# byte = file.read(1);

