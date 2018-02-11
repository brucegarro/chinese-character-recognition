import tensorflow as tf
import numpy as np

from load import load_hsk_100_data, reformat, IMG_SIZE

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def multinomial_logistic_regression_model():
	# Following this example https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
	(
		(train_data, train_labels),
		(valid_data, valid_labels),
		(test_data, test_labels),
	) = load_hsk_100_data()

	num_samples = train_data.shape[0] # 3000
	img_size = train_data.shape[1] # 224
	img_pixel_count = img_size**2 # 50176
	num_labels = train_labels.shape[1] # 100
	# Flatten image data since there will be no convolutions in this graph.
	# Pixels are treated indepent from their dimensions and channels
	train_data = train_data.reshape(train_data.shape[0], img_size**2)
	valid_data = valid_data.reshape(valid_data.shape[0], img_size**2)
	test_data = test_data.reshape(test_data.shape[0], img_size**2)

	print "Training set: %s, %s" % (train_data.shape, train_labels.shape)
	print "Validation set: %s, %s" % (valid_data.shape, valid_labels.shape)
	print "Test set: %s, %s" % (test_data.shape, test_labels.shape)

	# Parameters
	learning_rate = 0.01
	training_epochs = 25
	batch_size = 50
	display_steps = 1

	# tf Graph Input
	x = tf.placeholder(tf.float32, [None, img_pixel_count])
	y = tf.placeholder(tf.float32, [None, num_labels])

	# Set model weights
	W = tf.Variable(tf.zeros([img_pixel_count, num_labels]))
	b = tf.Variable(tf.zeros([num_labels]))

	# Construct Model
	pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

	# Cost function
	cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

	# Optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(training_epochs):
			avg_cost = 0.0 # aggregation variable
			total_batch = int(num_samples/batch_size)

			for i in range(total_batch):
				batch_idx = np.random.randint(num_samples, size=batch_size)
				batch_x, batch_y = train_data[batch_idx], train_labels[batch_idx]
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				avg_cost += c / total_batch

			if ((epoch+1) % display_steps) == 0:
				print "Epoch: %s, cost: %s" % ((epoch+1), avg_cost)

			# Test model
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			accuracy = (
				tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				  .eval({
					x: valid_data,
					y: valid_labels
				})
			)
			print "Test Accuracy: %s" % accuracy
			print ""


def multilayer_convolutional_model():
	(
		(train_data, train_labels),
		(valid_data, valid_labels),
		(test_data, test_labels),
	) = load_hsk_100_data()

	print "Training set: %s, %s" % (train_data.shape, train_labels.shape)
	print "Validation set: %s, %s" % (valid_data.shape, valid_labels.shape)
	print "Test set: %s, %s" % (test_data.shape, test_labels.shape)

	num_labels = train_labels.shape[1] # 100
	num_channels = 1


	batch_size = 16
	k = patch_size = 5
	depth = 16
	num_hidden = 100

	graph = tf.Graph()
	with graph.as_default():
		# input
		tf_train_data = tf.placeholder(tf.float32, shape=(batch_size, IMG_SIZE, IMG_SIZE, num_channels))
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		tf_valid_data = tf.constant(tf.float32, shape=(batch_size, num_labels))
		tf_test_data = tf.constant(test_data)

		# variables
		l1_weights = tf.Variable(tf.truncated_normal([k, k, num_channels, depth], stddev=0.1))
		l1_biases = tf.Variable(tf.zeros([depth]))
		l2_weights = tf.Variable(tf.truncated_normal([k, k, depth, depth], stddev=0.1))
		l2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
		size3 = ((IMG_SIZE - k + 1) // 2 - k + 1) // 2
		l3_weights = tf.Variable(tf.truncated_normal([size3 * size3 * depth, num_hidden], stddev=0.1))


		def model(data):
			k = 5 # kernal/patch size
			s = 2 # stride
			p = 2 # padding (same padding)
			# output size: o = i + (k -1) - 2p (for same padding)
			#              o = 224 + (5 -1) - 2*2 = 224

if __name__ == "__main__":
	multinomial_logistic_regression_model()
	# multilayer_convolutional_model()