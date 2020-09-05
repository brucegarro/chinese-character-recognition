# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from utils import conv_output_width, pool_output_width
from load.build_dataset import get_or_create_path_label_pickle
from load.utils import (
    create_image_and_label_data_set,
    get_class_label_map,
    train_valid_split,
)
from character_sets.hsk_50_characters import HSK_50_CLASS_LABELS

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def multi_conv_model(class_labels, target_class=0):
    path_label_data = get_or_create_path_label_pickle(class_labels)
    class_label_map = get_class_label_map(class_labels)
    image_dataset, labels_dataset = create_image_and_label_data_set(
        path_label_data,
        class_label_map,
        padding=16,
        padding_color_value=255,
    )
    (
        train_data,
        train_labels,
        valid_data,
        valid_labels,
    ) = train_valid_split(image_dataset, labels_dataset)

    num_samples = train_data.shape[0] # len(train_data)
    img_size = train_data.shape[1]
    img_pixel_count = img_size**2
    num_labels = train_labels.shape[1] # num_classes
    print "\nnum_labels: %s\n" % num_labels
    num_channels = 1

    print "Datasets"
    print "Training set: %s, %s" % (train_data.shape, train_labels.shape)
    print "Validation set: %s, %s" % (valid_data.shape, valid_labels.shape)
    print "num_samples: %s" % num_samples
    print "img_size: %s" % img_size
    print "img_pixel_count: %s" % img_pixel_count
    print "num_labels: %s" % num_labels
    print "num_channels: %s" % num_channels
    print ""

    # Hyperparameters
    dropout_rate = 0.1
    learning_rate = 0.001
    training_epochs = 25
    batch_size = 50
    display_steps = 1

    print "Hyperparameters"
    print "dropout_rate: %s" % dropout_rate
    print "learning_rate: %s" % learning_rate
    print "training_epochs: %s" % training_epochs
    print "batch_size: %s" % batch_size
    print ""

    # Parameters
    i1, k1, s1, p1 = (img_size, 10, 2, 4)
    o1 = conv_output_width(i1, k1, s1, p1)
    kernal_n1 = 32

    pool_k1, pool_s1 = (2, 2)
    pool_o1 = pool_output_width(o1, pool_k1, pool_s1)

    i2, k2, s2, p2 = (pool_o1, 8, 2, 3)
    o2 = conv_output_width(i2, k2, s2, p2)
    kernal_n2 = 64

    pool_k2, pool_s2 = (2, 2)
    pool_o2 = pool_output_width(o2, pool_k2, pool_s2)

    i3, k3, s3, p3 = (pool_o2, 5, 1, 1)
    o3 = conv_output_width(i3, k3, s3, p3)
    kernal_n3 = 128

    pool_k3, pool_s3 = (2, 2)
    pool_o3 = pool_output_width(o3, pool_k3, pool_s3)

    i4, k4, s4, p4 = (pool_o3, 3, 1, 2)
    o4 = conv_output_width(i4, k4, s4, p4)
    kernal_n4 = 256

    pool_k4, pool_s4 = (2, 2)
    pool_o4 = pool_output_width(o4, pool_k4, pool_s4)

    # fully_connected_n = 1024
    fully_connected_n = 100
    fc1_size = pool_o4 * pool_o4 * kernal_n4

    print "i1, k1, s1, p1: (%s, %s, %s, %s)" % (i1, k1, s1, p1)
    print "o1: %s" % o1
    print "kernal_n1: %s\n" % kernal_n1
    print "pool_k1, pool_s1: (%s, %s)" % (pool_k1, pool_s1)
    print "pool_o1: %s\n" % pool_o1

    print "i2, k2, s2, p2: (%s, %s, %s, %s)" % (i2, k2, s2, p2)
    print "o2: %s" % o2
    print "kernal_n2: %s\n" % kernal_n2
    print "pool_k2, pool_s2: (%s, %s)" % (pool_k2, pool_s2)
    print "pool_o2: %s\n" % pool_o2

    print "i3, k3, s3, p3: (%s, %s, %s, %s)" % (i3, k3, s3, p3)
    print "o3: %s" % o3
    print "kernal_n3: %s\n" % kernal_n3
    print "pool_k3, pool_s3: (%s, %s)" % (pool_k3, pool_s3)
    print "pool_o3: %s\n" % pool_o3

    print "i4, k4, s4, p4: (%s, %s, %s, %s)" % (i4, k4, s4, p4)
    print "o4: %s" % o4
    print "kernal_n4: %s\n" % kernal_n4
    print "pool_k4, pool_s4: (%s, %s)" % (pool_k4, pool_s4)
    print "pool_o4: %s\n" % pool_o4

    print "fc1_size: %s\n" % fc1_size
    
    tf_valid_data = tf.constant(valid_data)

    # X = tf.map_fn(lambda img: tf.image.per_image_standardization(img), 
    #         tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels)
    # ))
    X = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels))
    Y = tf.placeholder(tf.float32, shape=(None, num_labels))



    # w1 = tf.Variable(tf.truncated_normal([k1, k1, num_channels, kernal_n1], stddev=0.01))
    w1 = tf.Variable(tf.truncated_normal([k1, k1, num_channels, kernal_n1], stddev=0.01))
    b1 = tf.Variable(tf.zeros([kernal_n1]))
    
    # w2 = tf.Variable(tf.truncated_normal([k2, k2, kernal_n1, kernal_n2], stddev=0.01))
    w2 = tf.Variable(tf.truncated_normal([k2, k2, kernal_n1, kernal_n2], stddev=0.01))
    b2 = tf.Variable(tf.zeros([kernal_n2]))
    
    # w3 = tf.Variable(tf.truncated_normal([k3, k3, kernal_n2, kernal_n3], stddev=0.01))
    w3 = tf.Variable(tf.truncated_normal([k3, k3, kernal_n2, kernal_n3], stddev=0.01))
    b3 = tf.Variable(tf.zeros([kernal_n3]))
    
    # w4 = tf.Variable(tf.truncated_normal([k4, k4, kernal_n3, kernal_n4], stddev=0.01))
    w4 = tf.Variable(tf.truncated_normal([k4, k4, kernal_n3, kernal_n4], stddev=0.01))
    b4 = tf.Variable(tf.zeros([kernal_n4]))

    w5 = tf.Variable(tf.truncated_normal([ fc1_size, fully_connected_n ], stddev=0.01))
    b5 = tf.Variable(tf.constant(1.0, shape=[fully_connected_n]))
    
    w6 = tf.Variable(tf.truncated_normal([fully_connected_n, num_labels], stddev=0.01))
    b6 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    def model(X):
        # Convolutional Layers
        conv1 = tf.nn.conv2d(X, w1, [1, s1, s1, 1], padding="SAME")
        activation1 = tf.nn.relu(conv1 + b1)
        pool1 = tf.nn.max_pool(activation1, ksize=[1, pool_k1, pool_k1, 1], strides=[1, pool_s1, pool_s1, 1], padding="SAME")
        
        conv2 = tf.nn.conv2d(pool1, w2, [1, s2, s2, 1], padding="SAME")
        activation2 = tf.nn.relu(conv2 + b2)
        pool2 = tf.nn.max_pool(activation2, ksize=[1, pool_k2, pool_k2, 1], strides=[1, pool_s2, pool_s2, 1], padding="SAME")
        
        conv3 = tf.nn.conv2d(pool2, w3, [1, s3, s3, 1], padding="SAME")
        activation3 = tf.nn.relu(conv3 + b3)
        pool3 = tf.nn.max_pool(activation3, ksize=[1, pool_k3, pool_k3, 1], strides=[1, pool_s3, pool_s3, 1], padding="SAME")
        
        conv4 = tf.nn.conv2d(pool3, w4, [1, s4, s4, 1], padding="SAME")
        activation4 = tf.nn.relu(conv4 + b4)
        pool4 = tf.nn.max_pool(activation4, ksize=[1, pool_k4, pool_k4, 1], strides=[1, pool_s4, pool_s4, 1], padding="SAME")

        # Fully-connected Layer
        fc1 = tf.reshape(pool4, [-1, w5.get_shape().as_list()[0]])

        fc1 = tf.add(tf.matmul(fc1, w5), b5)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout_rate)

        # Output Layer
        output = tf.add(tf.matmul(fc1, w6), b6)
        
        return output

    # softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model(X), labels=Y)
    softmax = tf.nn.sigmoid_cross_entropy_with_logits(logits=model(X), labels=Y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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

            # Calculate Training Accuracy
            accuracy_batch_size = 50
            train_size = valid_data.shape[0]
            accuracy_batches = int(train_size/accuracy_batch_size)
            batch_accuracies = []
            # Randomized indexes
            train_acc_idx = np.random.randint(train_size, size=train_size)

            for i in range(accuracy_batches):
                start_idx = accuracy_batch_size * i
                end_idx = accuracy_batch_size * (i+1)
                acc_batch_idx = train_acc_idx[start_idx:end_idx]

                acc_batch_x = train_data[acc_batch_idx]
                acc_batch_y =  train_labels[acc_batch_idx]
                accuracy = (tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                  .eval({
                    X: acc_batch_x,
                    Y: acc_batch_y,
                }))

                batch_accuracies.append(accuracy)
                # import ipdb; ipdb.set_trace()
            train_acurracy = np.average(batch_accuracies)
            print "Training Accuracy: %s" % train_acurracy

            # Validate model
            accuracy_batch_size = 50
            validation_size = valid_data.shape[0]
            accuracy_batches = int(validation_size/accuracy_batch_size)
            
            batch_accuracies = []
            # Randomized indexes
            acc_idx = np.random.randint(validation_size, size=validation_size)
            
            for i in range(accuracy_batches):
                start_idx = accuracy_batch_size * i
                end_idx = accuracy_batch_size * (i+1)
                acc_batch_idx = acc_idx[start_idx:end_idx]

                acc_batch_x = valid_data[acc_batch_idx]
                acc_batch_y =  valid_labels[acc_batch_idx]
                accuracy = (tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                  .eval({
                    X: acc_batch_x,
                    Y: acc_batch_y
                }))
                batch_accuracies.append(accuracy)
            accuracy = np.average(batch_accuracies)

            print "Validation Accuracy: %s" % accuracy
            print ""

if __name__ == "__main__":
    CLASS_LABELS = HSK_50_CLASS_LABELS
    multi_conv_model(CLASS_LABELS)
