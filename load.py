import struct
import scipy.misc
import numpy as np
import glob
from collections import defaultdict
import os
from os.path import join

import local

side = 224 # must be a multiple of 32 to work with maxpooling in vgg16
num_classes = 60
full_data = defaultdict(list)

def write_image(tag_name, image, writer):
	# import ipdb; ipdb.set_trace()
	writer_path = join(local.COMPETITION_GNT_PATH, writer)
	if not os.path.exists(writer_path):
		os.mkdir(writer_path)
	output_path = join(writer_path, "%s.bmp" % tag_name)
	# write image to bmp
	scipy.misc.imsave(output_path, image)
	print output_path

def gnt_to_bmp(filepath, write_images=False):
	images = {}
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
			bytez = struct.unpack("{}B".format(height*width), raw_bytes)
			existing_labels = full_data.keys()
			if (tag_name in existing_labels) or (len(existing_labels) < num_classes):
				image = np.array(bytez).reshape(height, width)
				image = scipy.misc.imresize(image, (side, side))

				writer  = filepath.split("/")[-1].split(".")[0]
				if write_image:
					write_image(tag_name, image, writer)
				# scipy.misc.imsave("%s.bmp" % (tag_name), image)
				image = (image.astype(float) / 256) - 0.5 # normalize to [-0.5,0.5] to avoid saturation
				images[(writer, tag_name)] = image
				full_data[tag_name].append(image)
	print filepath
	return full_data

# writers = {
#    "writer1": {u"\u1004": [2 x 2 array of image intensities]
#    "writer2": ...
# }

def main():
	write_images = False
	writers = {}
	for name in [ f for f in os.listdir(local.COMPETITION_GNT_PATH) if f.endswith(".gnt") ]:
		filepath = join(local.COMPETITION_GNT_PATH, name)
		writer  = filepath.split("/")[-1].split(".")[0]
		writers[writer] = gnt_to_bmp(filepath, write_images=write_images)
	import ipdb; ipdb.set_trace()



if __name__=="__main__":
	main()