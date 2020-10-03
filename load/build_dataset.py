# -*- coding: utf-8 -*-
import os
from os.path import join
import tensorflow as tf
from six.moves import cPickle as pickle

import settings
from load.os_utils import (
    get_filepaths_for_class_labels,
    get_path_label_pickle_path,
    pickle_path_label_map,
    open_path_label_map,
    get_all_classes_with_counts_in_filesystem,
)
from load.utils import (
    create_image_and_label_data_set,
    get_class_label_map,
    train_valid_split,
)
from character_sets.hsk_10_characters import HSK_10_CLASS_LABELS
from character_sets.hsk_50_characters import HSK_50_CLASS_LABELS
from character_sets.hsk_100_characters import HSK_100_CLASS_LABELS


def map_img_path_to_array(image_path):
    image = tf.io.decode_bmp(tf.read_file(image_path))
    return image

def map_fn(image_path, label):
    return map_img_path_to_array(image_path), label

def build_dataset_for_class_labels(image_paths, labels):
    # Create Tensorflow Dataset out of image and labels dataset
    image_paths_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)
    label_tensor = tf.convert_to_tensor(labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, label_tensor))
    dataset = dataset.map(map_fn, num_parallel_calls=2)

    return dataset


def get_or_create_path_label_pickle(class_labels):
    pickle_path = get_path_label_pickle_path(len(class_labels))

    if not os.path.exists(pickle_path):
        path_generator = get_filepaths_for_class_labels(
            class_labels,
            source_paths=settings.TRAIN_SOURCE_PATHS
        )
        pickle_path_label_map(path_generator, pickle_path)

    path_label_data = open_path_label_map(pickle_path)
    return path_label_data

def get_or_create_class_label_count_pickle():
    filename = "class_label_counts.pickle"
    class_label_count_pickle_path = join(settings.DATA_PATH, filename)
    
    if os.path.exists(class_label_count_pickle_path):
        # Open existing file
        with open(class_label_count_pickle_path, "rb") as f:
            data = pickle.load(f)
    else:
        # Create a file if it doesn't exist
        with open(class_label_count_pickle_path, "wb") as f:
            class_label_counts = get_all_classes_with_counts_in_filesystem()
            data = pickle.dump(class_label_counts, f, pickle.HIGHEST_PROTOCOL)
    return data

def build_in_memory_dataset(class_labels):
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

    return {
        "train_data": train_data,
        "train_labels": train_labels,
        "valid_data": valid_data,
        "valid_labels": valid_labels,
    }

if __name__ == "__main__":
    # Get list of image files and corresponding labels
    CLASS_LABELS = HSK_50_CLASS_LABELS
    path_label_data = get_or_create_path_label_pickle(CLASS_LABELS)
    
    # Create pickle with class label count
    # get_or_create_class_label_count_pickle()
