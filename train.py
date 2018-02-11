import tensorflow as tf

from load import load_hsk_data, reformat

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

if __name__ == "main":
	data = load_hsk_data()

	train_data, train_labels = reformat(data["train_data"], data["train_labels"])
	valid_data, valid_labels = reformat(data["valid_data"], data["valid_labels"])
	test_data, test_labels = reformat(data["test_data"], data["test_labels"])

	print "Training set: %s, %s" % (train_data.shape, train_labels.shape)
	print "Validation set: %s, %s" % (valid_data.shape, valid_labels.shape)
	print "Test set: %s, %s" % (test_data.shape, test_labels.shape)

	i = img_size = 224
	num_labels = train_labels.shape[1] # 100
	num_channels = 1


	batch_size = 16
	k = patch_size = 5
	depth = 16
	num_hidden = 100

	graph = tf.Graph()

	with graph.as_default():
		# input
		tf_train_data = tf.placeholder(tf.float32, shape=(batch_size, i, i, num_channels))
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		tf_valid_data = tf.constant(tf.float32, shape=(batch_size, num_labels))
		tf_test_data = tf.constant(test_data)

		# variables
		l1_weights = tf.Variable(tf.truncated_normal([k, k, num_channels, depth], stddev=0.1))
		l1_biases = tf.Variable(tf.zeros([depth]))
		l2_weights = tf.Variable(tf.truncated_normal([k, k, depth, depth], stddev=0.1))
		l2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
		size3 = ((i - k + 1) // 2 - k + 1) // 2
		l3_weights = tf.Variable(tf.truncated_normal([size3 * size3 * depth, num_hidden], stddev=0.1))


		def model(data):
			i = 224 # image size
			k = 5 # kernal/patch size
			s = 2 # stride
			p = 2 # padding (same padding)
			# output size: o = i + (k -1) - 2p (for same padding)
			#              o = 224 + (5 -1) - 2*2 = 224