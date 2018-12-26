import tensorflow as tf
import numpy as np

from load import load_hsk_100_data, reformat, IMG_SIZE
from utils import conv_output_width, pool_output_width


def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def double_hidden_layer_convolutional_model():
	"""
	Using this example as guidance: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb
	"""
	(
		(train_data, train_labels),
		(valid_data, valid_labels),
		(test_data, test_labels),
	) = load_hsk_100_data()

	num_samples = train_data.shape[0] # 3000
	img_size = train_data.shape[1] # 224
	img_pixel_count = img_size**2 # 50176
	num_labels = train_labels.shape[1] # 100
	num_channels = 1

	# Hyperparameters
	dropout_rate = 0.75
	learning_rate = 0.01
	training_epochs = 25
	batch_size = 16
	display_steps = 1

	print "Training set: %s, %s" % (train_data.shape, train_labels.shape)
	print "Validation set: %s, %s" % (valid_data.shape, valid_labels.shape)
	print "Test set: %s, %s" % (test_data.shape, test_labels.shape)
	print "Sample size: %s" % num_samples
	print "Batch size: %s" % batch_size

	# Parameters
	i1, k1, s1, p1 = (img_size, 3, 1, 1)
	o1 = conv_output_width(i1, k1, s1, p1)
	kernal_n1 = 32

	pool_k1, pool_s1 = (2, 2)
	pool_o1 = pool_output_width(o1, pool_k1, pool_s1)

	i2, k2, s2, p2 = (pool_o1, 3, 1, 1)
	o2 = conv_output_width(i2, k2, s2, p2)
	kernal_n2 = 64

	pool_k2, pool_s2 = (2, 2)
	pool_o2 = pool_output_width(o2, pool_k2, pool_s2)

	fully_connected_n = 1024
	
	tf_valid_data = tf.constant(valid_data)
	tf_test_data = tf.constant(test_data)

	X = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, num_channels))
	Y = tf.placeholder(tf.float32, shape=(None, num_labels))

	w1 = tf.Variable(tf.truncated_normal([k1, k1, num_channels, kernal_n1], stddev=0.01))
	b1 = tf.Variable(tf.zeros([kernal_n1]))
	
	# TODO: Why does o2*2 work here?
	w2 = tf.Variable(tf.truncated_normal([k2, k2, kernal_n1, kernal_n2], stddev=0.01))
	b2 = tf.Variable(tf.zeros([kernal_n2]))

	w3 = tf.Variable(tf.truncated_normal([ pool_o2 * pool_o2 * kernal_n2, fully_connected_n ], stddev=0.01))
	b3 = tf.Variable(tf.constant(1.0, shape=[fully_connected_n]))
	
	w4 = tf.Variable(tf.truncated_normal([fully_connected_n, num_labels], stddev=0.01))
	b4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

	def model(X):
		# Convolutional Layers
		conv1 = tf.nn.conv2d(X, w1, [1, s1, s1, 1], padding="SAME")
		activation1 = tf.nn.relu(conv1 + b1)
		pool1 = tf.nn.max_pool(activation1, ksize=[1, pool_k1, pool_k1, 1], strides=[1, pool_s1, pool_s1, 1], padding="SAME")
		
		conv2 = tf.nn.conv2d(pool1, w2, [1, s2, s2, 1], padding="SAME")
		activation2 = tf.nn.relu(conv2 + b2)
		pool2 = tf.nn.max_pool(activation2, ksize=[1, pool_k2, pool_k2, 1], strides=[1, pool_s2, pool_s2, 1], padding="SAME")

		# Fully-connected Layer
		fc1 = tf.reshape(pool2, [-1, w3.get_shape().as_list()[0]])

		fc1 = tf.add(tf.matmul(fc1, w3), b3)
		fc1 = tf.nn.relu(fc1)
		fc1 = tf.nn.dropout(fc1, dropout_rate)

		# Output Layer
		output = tf.add(tf.matmul(fc1, w4), b4)
		
		return output

	softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model(X), labels=Y)
	cost = tf.reduce_mean(softmax)
	optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

	# Runtime configurations
	init = tf.global_variables_initializer()

	options = tf.RunOptions()
	options.output_partition_graphs = True
	options.report_tensor_allocations_upon_oom = True
	options.trace_level = tf.RunOptions.FULL_TRACE

	gpu_usage_limit = 0.75
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage_limit)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		metadata = tf.RunMetadata()
		sess.run(init, options=options)
		
		for epoch in range(training_epochs):
				avg_cost = 0.0 # aggregation variable
				total_batch = int(num_samples/batch_size)

				for i in range(total_batch):
					batch_idx = np.random.randint(num_samples, size=batch_size)
					batch_x, batch_y = train_data[batch_idx], train_labels[batch_idx]
					_, c = sess.run([optimizer, cost], feed_dict={
						X: batch_x,
						Y: batch_y
					})
					avg_cost += c / total_batch

				if ((epoch+1) % display_steps) == 0:
					print "Epoch: %s, cost: %s" % ((epoch+1), avg_cost)

				correct_predictions = tf.equal(tf.argmax(softmax, -1), tf.argmax(Y, -1))

				# Validate model
				accuracy_batch_size = 100
				validation_size = valid_data.shape[0]
				accuracy_batches = int(validation_size/accuracy_batch_size)
				
				batch_accuracies = []
				for i in range(accuracy_batches):
					acc_batch_idx = np.random.randint(validation_size, size=accuracy_batch_size)
					acc_batch_x, acc_batch_y = valid_data[acc_batch_idx], valid_labels[acc_batch_idx]
					batch_accuracies.append(
						tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
						  .eval({
							X: acc_batch_x,
							Y: acc_batch_y
						})
					)
				accuracy = np.average(batch_accuracies)

				print "Validation Accuracy: %s" % accuracy
				print ""

if __name__ == "__main__":
	double_hidden_layer_convolutional_model()
