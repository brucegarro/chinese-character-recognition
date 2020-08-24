# -*- coding: utf-8 -*-
import os
import tensorflow as tf

import settings
from load.os_utils import (
    get_filepaths_for_class_label,
    get_filepaths_for_class_labels,
    get_path_label_pickle_path,
    pickle_path_label_map,
    open_path_label_map,
)

CLASS_LABELS = [
    "一",
    "三",
    "上",
    "下",
    "不",
    "与",
    "丑",
    "丙",
    "丛",
    "东",
]

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


def map_img_path_to_array(image_path):
    image = tf.io.decode_bmp(tf.read_file(image_path))
    return image

def map_fn(image_path, label):
    return map_img_path_to_array(image_path), label

def build_dataset_for_class_labels(image_paths, labels):
    # https://stackoverflow.com/questions/44416764/loading-folders-of-images-in-tensorflow
    image_paths_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)
    label_tensor = tf.convert_to_tensor(labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, label_tensor))
    dataset = dataset.map(map_fn, num_parallel_calls=2)

    return dataset

if __name__ == "__main__":
    class_labels = CLASS_LABELS

    # Get list of image files and corresponding labels
    path_label_data = get_or_create_path_label_pickle(class_labels)
    # for path, label in path_label_data:
    #     print path, label

    image_paths = [ str(path) for path, label in path_label_data ]
    labels = [ label for path, label in path_label_data ]