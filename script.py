import struct
import scipy.misc
import numpy
import glob
from collections import defaultdict
import pdb
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
		label = struct.unpack(">H", f.read(2))[0]
		label -= 0xb0a1 # CASIA labels start at 0xb0a1
		width = struct.unpack("<H", f.read(2))[0]
		height = struct.unpack("<H", f.read(2))[0]
		raw_bytes = f.read(height*width)
		bytes = struct.unpack("{}B".format(height*width), raw_bytes)
		existing_labels = full_data.keys()
		if (label in existing_labels) or (len(existing_labels) < num_classes):
			image = numpy.array(bytes).reshape(height, width)
			image = scipy.misc.imresize(image, (side, side))
			if (1==0):
				scipy.misc.imsave("%d.bmp" % (counter), image)
			image = (image.astype(float) / 256) - 0.5 # normalize to [-0.5,0.5] to avoid saturation
			full_data[label].append(image)
			counter = counter + 1
	f.close()

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

