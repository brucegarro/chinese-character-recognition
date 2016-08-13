import struct
import scipy.misc
import numpy
import pdb
import glob
from collections import defaultdict
from random import shuffle

# cd ~/Desktop/chinese-character-recognition/competition-gnt

# GET THE DATA
#full_data = numpy.load('dic.py.npy').item()
#label_to_character = {0: '啊', 1: '阿', 2: '埃', 3: '挨', 4: '哎', 5: '唉', 6: '哀', 7: '皑', 8: '癌', 9: '蔼', 10: '矮', 11: '艾', 12: '碍', 13: '爱', 14: '隘', 15: '鞍', 16: '氨', 17: '安', 18: '俺', 19: '按', 20: '暗', 21: '岸', 22: '胺', 23: '案', 24: '肮', 25: '昂', 26: '盎', 27: '凹', 28: '敖', 29: '熬', 30: '翱', 31: '袄', 32: '傲', 33: '奥', 34: '懊', 35: '澳', 36: '芭', 37: '捌', 38: '扒', 39: '叭', 40: '吧', 41: '笆', 42: '八', 43: '疤', 44: '巴', 45: '拔', 46: '跋', 47: '靶', 48: '把', 49: '耙', 50: '坝', 51: '霸', 52: '罢', 53: '爸', 54: '白', 55: '柏', 56: '百', 57: '摆', 58: '佰', 59: '败', 60: '拜'}

side = 224 # must be a multiple of 32 to work with maxpooling in vgg16
full_data = defaultdict(list)
label_to_character = {}
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
		label = struct.unpack(">H", tag_name)[0]
		label -= 0xb0a1 # CASIA labels start at 0xb0a1
		tag_name = tag_name.decode("gb2312")
		label_to_character[label] = tag_name
		width = struct.unpack("<H", f.read(2))[0]
		height = struct.unpack("<H", f.read(2))[0]
		bytes = struct.unpack("{}B".format(height*width), f.read(height*width))
		existing_labels = full_data.keys()
		if (label in existing_labels) or (len(existing_labels) < num_classes):
			image = numpy.array(bytes).reshape(height, width)
			image = scipy.misc.imresize(image, (side, side))
			#scipy.misc.imsave("%s%d.bmp" % (tag_name, counter), image)
			image = (image.astype(float) / 256) # - 0.5 # normalize to [-0.5,0.5] to avoid saturation
			full_data[label].append(image)
			counter = counter + 1
	f.close()

training_data = defaultdict(list)
test_data = defaultdict(list)

for key, values in full_data.items():
	shuffle(values)
	training_data[key] = []
	test_data[key] = []
	for value in values[0:50]:
		training_data[key].append(numpy.array(value).flatten())
	for value in values[50:]:
		test_data[key].append(numpy.array(value).flatten())


# numpy.save('dic.py.npy', full_data)

# structure of full_data -- dictionary with 60 keys,
# each of which is a character pointing to 60 examples
# each example is a 224x224 matrix


# TENSORFLOW TIME!!!!
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 50176])
y_ = tf.placeholder(tf.float32, shape=[None, 4])

W = tf.Variable(tf.zeros([50176, 4]))
b = tf.Variable(tf.zeros([4]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
	for key in [0,1,2,3]:
		batch_xs = training_data.get(key)
		batch_ys = numpy.tile(numpy.eye(4)[key],(len(batch_xs),1))
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1));
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32));

for key in [0,1,2,3]:
	test_xs = test_data.get(key);
	test_ys = numpy.tile(numpy.eye(4)[key],(len(test_xs),1))
	print(sess.run(accuracy, feed_dict={x: test_xs, y_:test_ys}))

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