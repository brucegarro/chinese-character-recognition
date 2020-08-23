# -*- coding: utf-8 -*-
import os
import tensorflow as tf

import settings
from os_utils import (
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

def build_dataset_for_class_labels(class_labels):
    pass

if __name__ == "__main__":
    class_labels = CLASS_LABELS

    path_label_data = get_or_create_path_label_pickle(class_labels)
    for path, label in path_label_data:
        print path, label