import tensorflow as tf
import numpy as np

from load import load_hsk_100_data, reformat, IMG_SIZE

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def double_hidden_layer_convolutional_model():
	(
		(train_data, train_labels),
		(valid_data, valid_labels),
		(test_data, test_labels),
	) = load_hsk_100_data()

	print "Training set: %s, %s" % (train_data.shape, train_labels.shape)
	print "Validation set: %s, %s" % (valid_data.shape, valid_labels.shape)
	print "Test set: %s, %s" % (test_data.shape, test_labels.shape)
	
	num_samples = train_data.shape[0] # 3000
	i = img_size = train_data.shape[1] # 224
	img_pixel_count = img_size**2 # 50176
	num_labels = train_labels.shape[1] # 100
	num_labels = train_labels.shape[1] # 100
	num_channels = 1

	# Parameters
	k = patch_size = 4
	s = 2 # x, y stride
	stride = [1, s, s, 1]
	p = zero_padding = 1 # SAME padding
	depth = num_hidden = int(((i - 2*p + k) / float(s)) + 1) # 114 for same p=1, k=4
	assert(depth % 1 == 0)
	depth = int(depth)
	fully_connected_n = 50
	
	learning_rate = 0.01
	training_epochs = 25
	batch_size = 16
	display_steps = 1

	tf_train_data = tf.placeholder(tf.float32, shape=(batch_size, IMG_SIZE, IMG_SIZE, num_channels))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_data = tf.constant(valid_data)
	tf_test_data = tf.constant(test_data)

	w1 = tf.Variable(tf.truncated_normal([k, k, num_channels, depth], stddev=0.01))
	b1 = tf.Variable(tf.zeros([depth]))
	w2 = tf.Variable(tf.truncated_normal([k, k, depth, depth*2], stddev=0.01))
	b2 = tf.Variable(tf.zeros([depth*2]))
	input_size3 = img_pixel_count // 4 * img_pixel_count // 4 * depth * 2 * 2
	w3 = tf.Variable(tf.truncated_normal([ input_size3, fully_connected_n ], stddev=0.01))
	b3 = tf.Variable(tf.constant(1.0, shape=[fully_connected_n]))
	w4 = tf.Variable(tf.truncated_normal([fully_connected_n, num_labels], stddev=0.01))
	b4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

	def model(data):
		# Convolutional Layers
		conv1 = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding="SAME")
		pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
		l1 = tf.nn.relu(pool1 + b1)
		conv2 = tf.nn.conv2d(l1, w2, [1, 1, 1, 1], padding="SAME")
		pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
		l2 = tf.nn.relu(pool2 + b2)
		
		# import ipdb; ipdb.set_trace()
		# Fully-connected Layer
		reshape = tf.reshape(l2, (-1, input_size3))
		l3 = tf.nn.relu(tf.matmul(reshape, w3) + b3)

		# Output Layer
		l4 = tf.matmul(l3, w4) + b4
		return l4

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model(tf_test_data), labels=tf_train_labels))
	optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
	train_prediction = tf.nn.softmax(model(tf_train_data))
	valid_prediction = tf.nn.softmax(model(tf_valid_data))
	test_prediction = tf.nn.softmax(model(tf_test_data))

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		
		for epoch in range(training_epochs):
				avg_cost = 0.0 # aggregation variable
				total_batch = int(num_samples/batch_size)

				for i in range(total_batch):
					batch_idx = np.random.randint(num_samples, size=batch_size)
					batch_x, batch_y = train_data[batch_idx], train_labels[batch_idx]
					_, c = sess.run([optimizer, cost], feed_dict={
						tf_train_data: batch_x,
						tf_train_labels: batch_y
					})
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
				print "Validation Accuracy: %s" % accuracy
				print ""

if __name__ == "__main__":
	double_hidden_layer_convolutional_model()
