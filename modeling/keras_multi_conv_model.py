# -*- coding: utf-8 -*-
import tensorflow as tf
import keras.backend
from keras.callbacks import TensorBoard
import numpy as np

from utils import conv_output_width, pool_output_width
from load.build_dataset import build_in_memory_dataset, build_dataset_for_class_labels
from character_sets.hsk_50_characters import HSK_50_CLASS_LABELS
from character_sets.hsk_10_characters import HSK_10_CLASS_LABELS

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def multi_conv_model(dataset, target_class=0):
    train_data = dataset["train_data"]
    train_labels = dataset["train_labels"]
    valid_data = dataset["valid_data"]
    valid_labels = dataset["valid_labels"]

    num_samples = train_data.shape[0] # len(train_data)
    img_size = train_data.shape[1]
    img_pixel_count = img_size**2
    num_labels = len(train_labels) # num_classes
    num_channels = 1
    print "\nnum_labels: %s\n" % num_labels

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
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 50

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


    # Define model
    model = tf.keras.Sequential([
        # Convolutional Layer 1
        tf.keras.layers.Conv2D(
            filters=kernal_n1, kernel_size=(k1, k1), strides=(s1, s1),
            padding="SAME", activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=kernal_n1, strides=(pool_s1, pool_s1), padding="SAME"),
        # Convolutional Layer 2
        tf.keras.layers.Conv2D(
            filters=kernal_n2, kernel_size=(k2, k2), strides=(s2, s2),
            padding="SAME", activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=kernal_n2, strides=(pool_s2, pool_s2), padding="SAME"
        ),
        # Convolutional Layer 3
        tf.keras.layers.Conv2D(
            filters=kernal_n3, kernel_size=(k3, k3), strides=(s3, s3),
            padding="SAME", activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=kernal_n3, strides=(pool_s3, pool_s3), padding="SAME"),
        # Convolutional Layer 4
        tf.keras.layers.Conv2D(
            filters=kernal_n4, kernel_size=(k4, k4), strides=(s4, s4),
            padding="SAME", activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=kernal_n4, strides=(pool_s4, pool_s4), padding="SAME"
        ),
        # Flatten -> Fully-Connected Layer -> Dropout
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=fully_connected_n, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        # Output Layer
        tf.keras.layers.Dense(
            units=num_labels,
            activation="softmax",
        ),
    ])
    
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    tensor_board = TensorBoard(log_dir="./logs", histogram_freq=1)

    # Runtime configurations
    # init = tf.global_variables_initializer()

    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.report_tensor_allocations_upon_oom = True
    options.trace_level = tf.RunOptions.FULL_TRACE

    gpu_usage_limit = 0.75
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage_limit)

    # Convert label format from one-hot to label index for sparse categorical cross-entropy
    train_labels = np.array(map(np.argmax, train_labels))
    valid_labels = np.array(map(np.argmax, valid_labels))
    

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.fit(
            train_data,
            train_labels,
            validation_data=(valid_data, valid_labels),
            batch_size=batch_size,
            epochs=training_epochs,
        )

if __name__ == "__main__":
    # CLASS_LABELS = HSK_10_CLASS_LABELS
    CLASS_LABELS = HSK_50_CLASS_LABELS
    dataset = build_in_memory_dataset(CLASS_LABELS)
    multi_conv_model(dataset)
