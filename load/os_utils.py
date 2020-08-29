import re
import itertools
import os
from os.path import join
from pathlib import Path
from six.moves import cPickle as pickle

import settings

def get_all_gnt_filepaths_in_folderpath(folderpath):
    gnt_names = [ name for name in os.listdir(folderpath) if name.endswith(".gnt") ]
    gnt_filepaths = [ join(folderpath, name) for name in gnt_names ]
    return gnt_filepaths

def get_filepaths_for_class_label(class_label, source_paths):
    generators = ([ 
        Path(base_path).rglob("*/%s.bmp" % class_label) for base_path in source_paths
    ])
    for generator in generators:
        for path in generator:
            yield path, class_label

def get_filepaths_for_class_labels(class_labels, source_paths=settings.GNT_SOURCE_PATHS):
    class_label_generators = [
        get_filepaths_for_class_label(class_label, source_paths) for class_label in class_labels
    ]
    path_generator = itertools.chain(*class_label_generators)
    return path_generator

def get_path_label_pickle_path(num_classes):
    filename = "path_label_mapping_for_%s_labels.pickle" % num_classes
    pickle_path = join(settings.DATA_PATH, filename)
    return pickle_path

def pickle_path_label_map(path_generator, pickle_path):
    print "Creating: %s" % pickle_path
    with open(pickle_path, "wb") as f:
        pickle.dump(list(path_generator), f, pickle.HIGHEST_PROTOCOL)
        print "Created: %s\n" % pickle_path

def open_path_label_map(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def is_one_chinese_character(text, regex=re.compile(ur'[\u4e00-\u9fff]+')):
    utf_8_text = text.decode("utf-8")
    return (len(utf_8_text) == 1) and bool(regex.match(utf_8_text))

def get_all_classes_with_counts_in_filesystem(source_paths=settings.GNT_SOURCE_PATHS):
    # Get labels from the names on the bmp files
    class_label_counts = {}

    for path, _ in get_filepaths_for_class_labels(["*"]):
        if is_one_chinese_character(path.stem):
            class_label = path.stem
            if class_label not in class_label_counts:
                class_label_counts[class_label] = 1
            else:
                class_label_counts[class_label] += 1

    return class_label_counts
