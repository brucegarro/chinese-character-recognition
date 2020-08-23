"""
Produces folders with bmps generated of the HWDB1.1 Chinese character dataset of gnt files
"""

import struct
import scipy.misc
import numpy as np
import glob
from collections import defaultdict
import os
from os.path import join
from six.moves import cPickle as pickle
import tensorflow as tf

import settings
from hsk import vocab

IMG_SIZE = 224 # must be a multiple of 32 to work with maxpooling in vgg16

TRAIN_SET_SIZE = 0.8
VALID_SET_SIZE = 0.2
TEST_SET_SIZE = 0.0

DEFAULT_NUMBER_OF_CLASSES = 100
DEFAULT_HSK_LEVELS = (1, 2, 3)


def open_image_as_array(filepath, normalize=False):
    with open(filepath, "rb") as f:
        img = scipy.misc.imread(f, flatten=True)

        if normalize:
            # If normalize is True, normalize pixel range to to range [-0.5, 0.5]
            for i in range(len(img)):
                for j in range(len(img[0])):
                    img[i][j] = (img[i][j]/255.0) - 0.5

    return img

def get_competition_author_directory_names():
    """
    In the COMPETITION_GNT_PATH get the bmp directory names. These directory names also server as the author names

    Returns
    -------
    list: [str] - directory names (author names) e.g "C001-f-f"
    """
    return sorted([ f for f in os.listdir(settings.COMPETITION_GNT_PATH) if (f.startswith("C") and f.endswith("f-f")) ])

def get_bmp_path_for_directory_name(author_name, classes):
    """
    For a given author and set of classes, return the full paths of the bmp files.

    Returns
    -------
    list: [str] - full file paths
    """
    bmps_directory = join(settings.COMPETITION_GNT_PATH, author_name)
    bmps_names = [ bmp_name for bmp_name in os.listdir(bmps_directory) if bmp_name.endswith(".bmp") and bmp_name.strip(".bmp") in classes ]
    return sorted(bmps_names)

def get_all_classes():
    """
    Not all writers have written examples for every classes so
    determine the classes which are present for all writers

    Returns
    -------
    list: [str] - sorted list of authors
    """
    classes = None
    author_directory_names = get_competition_author_directory_names()
    for name in author_directory_names:
        filepath = join(settings.COMPETITION_GNT_PATH, name.strip(".gnt"))
        class_names = set([ bmp_name.strip(".bmp") for bmp_name in os.listdir(filepath) if bmp_name.endswith(".bmp") ])
        if classes == None:
            classes = class_names
        else:
            classes &= class_names
    
    sorted_classes = sorted(list(classes))

    return sorted_classes

def filter_classes_by_hsk_level(classes, hsk_levels=(1,2,3,4,5,6)):
    """
    Filters Chinese character classes by whether they are part of HSK vocabulary lists

    Returns
    -------
    list: [str] - classes (chinese characters)
    """
    return [ cl for cl in classes if vocab.get(cl) in hsk_levels ]

def get_image_arrays_for_author_directory_names(author_directory_names, classes):
    """
    Returns
    -------
    img_data: Images data for all authors and classes
        type: np.ndarray
        dimensions: (number_of_authors, num_classes, img_size, img_size)

    labels: Class labels for all authors and classes
        type: np.ndarray
        dimensions: (number_of_authors, num_classes)
    """
    class_labels = {label: i for i, label in enumerate(classes)}

    num_classes = len(classes)
    number_of_authors = len(author_directory_names)
    data_set_size = int(num_classes * number_of_authors)

    img_data = np.ndarray((number_of_authors, num_classes, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    labels = np.ndarray((number_of_authors, num_classes), dtype=np.int32)

    for author_i, author_name in enumerate(author_directory_names):
        print "\nauthor_name: %s" % author_name
        bmp_directory = join(settings.COMPETITION_GNT_PATH, author_name)
        bmp_names = get_bmp_path_for_directory_name(author_name, classes)

        for bmp_name in bmp_names:
            bmp_path = join(settings.COMPETITION_GNT_PATH, author_name, bmp_name)
            class_char = bmp_name.strip(".bmp")
            img = open_image_as_array(bmp_path)
            # img_as_array, label, author_name, (class_char optional)
            label_i = class_labels[class_char]

            print "author_name, author_i, bmp_name, label_i: (%s, %s, %s, %s)" % (
                author_name,
                author_i,
                bmp_name.strip(".bmp"),
                label_i,
            )

            np.copyto(img_data[author_i][label_i], img)
            labels[author_i][label_i] = label_i
    return img_data, labels


def bmps_to_pickle(num_classes=DEFAULT_NUMBER_OF_CLASSES, hsk_levels=DEFAULT_HSK_LEVELS):
    all_classes = get_all_classes()
    classes = filter_classes_by_hsk_level(all_classes)[:num_classes]

    # Get data file system locations
    author_directory_names = get_competition_author_directory_names()

    # Covert images to arrays
    img_data, labels = get_image_arrays_for_author_directory_names(author_directory_names, classes)

    # Training set split code
    number_of_authors, num_classes, img_size, _ = img_data.shape

    train_size = int(number_of_authors*num_classes*TRAIN_SET_SIZE)
    valid_size = int(number_of_authors*num_classes*VALID_SET_SIZE)
    test_size = int(number_of_authors*num_classes*TEST_SET_SIZE)

    train_set_upper_bound = (TRAIN_SET_SIZE*number_of_authors)
    valid_set_upper_bound = ((TRAIN_SET_SIZE+VALID_SET_SIZE)*number_of_authors)

    train_data = np.ndarray((train_size, img_size, img_size), dtype=np.float32)
    valid_data = np.ndarray((valid_size, img_size, img_size), dtype=np.float32)
    test_data = np.ndarray((test_size, img_size, img_size), dtype=np.float32)
    train_labels = np.ndarray(train_size, dtype=np.int32)
    valid_labels = np.ndarray(valid_size, dtype=np.int32)
    test_labels = np.ndarray(test_size, dtype=np.int32)

    random_author_indexes = list(np.arange(number_of_authors))
    np.random.seed(0)
    np.random.shuffle(random_author_indexes)

    train_i = valid_i = test_i = 0

    for author_randomizer_i, author_i in enumerate(random_author_indexes):
        author_name = author_directory_names[author_i]
        print "\nauthor_name: %s" % author_name
        training_chars = []
        valid_chars = []
        test_chars = []

        bmps_names = get_bmp_path_for_directory_name(author_name, classes)

        for label_i, bmp_name in enumerate(bmps_names):
            bmp_path = join(settings.COMPETITION_GNT_PATH, author_name, bmp_name)
            class_char = bmp_name.strip(".bmp")
            img = img_data[author_i][label_i]
            label = labels[author_i][label_i]

            print "author_name, author_i, bmp_name, label: (%s, %s, %s, %s)" % (
                author_name,
                author_i,
                bmp_name.strip(".bmp"),
                label,
            )

            if author_randomizer_i < train_set_upper_bound:
                training_chars.append(class_char)
                np.copyto(train_data[train_i], img)
                train_labels[train_i] = label
                train_i += 1
            elif train_set_upper_bound <= author_randomizer_i < valid_set_upper_bound:
                valid_chars.append(class_char)
                np.copyto(valid_data[valid_i], img)
                valid_labels[valid_i] = label
                valid_i += 1
            else:
                test_chars.append(class_char)
                np.copyto(test_data[test_i], img)
                test_labels[test_i] = label
                test_i += 1
        print u"training_chars: %s".encode("utf-8") % " ".join(sorted(training_chars))
        print u"valid_chars: %s".encode("utf-8") % " ".join(sorted(valid_chars))
        print u"test_chars: %s".encode("utf-8") % " ".join(sorted(test_chars))

    assert train_i == train_size
    assert valid_i == valid_size
    assert test_i == test_size

    print "train_labels: %s" % train_labels
    print "valid_labels: %s" % valid_labels
    print "test_labels: %s" % test_labels
    output = {
        "train_data": train_data,
        "train_labels": train_labels,
        "valid_data": valid_data,
        "valid_labels": valid_labels,
        "test_data": test_data,
        "test_labels": test_labels,
    }

    # TODO: Remove the constant HSK_100_PICKLE_PATH and create this path within this function.
    # below is a hack to get the name of the output path to reflect the correct number of classes.
    output_path = join(settings.DATA_PATH, (settings.HSK_FILENAME % num_classes))

    f = open(output_path, "wb")
    pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
    print "pickle written to: %s" % output_path
    f.close()

def reformat_img_data(img_data, num_channels=1, padding=16):
    """
    Format
    -------
        img_data: (N, IMG_SIZE, IMG_SIZE, num_channels)
    """
    _, x_size, y_size = img_data.shape

    args = (-1, x_size, y_size, num_channels)
    img_data = img_data.reshape(args).astype(np.float32)
    
    # Add white padding to images
    white_value = 0.5
    padding_dim = (
        (0, 0), # number of samples
        (padding, padding), # X dim
        (padding, padding), # y dim
        (0, 0) # channel dim
    )
    img_data = np.pad(img_data, padding_dim, constant_values=white_value, mode="constant")
    return img_data

def reformat_labels(labels):
    num_labels = len(set(labels))
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels

def reformat_to_binary_labels(labels, target_class):
    # Convert target label to 1; Convert all others to 0
    labels = np.equal(labels, target_class).astype(np.float32)
    # Resphare into N by 2 matrix
    labels =(np.arange(2) == labels[:,None]).astype(np.float32)
    return labels

def reformat(data, labels, num_channels=1, padding=16):
    """
    Format
    -------
        data: (N, IMG_SIZE, IMG_SIZE, num_channels)
        labels: (N, num_labels)
    """
    data = reformat_img_data(data, num_channels=num_channels, padding=padding)
    labels = reformat_labels(labels)

    return data, labels

def open_pickle(num_classes):
    filename = "hsk_%s_dataset.pickle" % num_classes
    pickle_path = join(settings.DATA_PATH, filename)
    with open(pickle_path, "rb") as f:
         data = pickle.load(f)
    return data

def load_hsk_data(num_classes):
    """
    Format
    ------
        train_data, train_label: (3000, 224, 224, 1), (3000, 100)
        valid_data, valid_label: (1800, 224, 224, 1), (1800, 100)
        test_data, test_label: (1200, 224, 224, 1), (1200, 100)
    """
    data = open_pickle(num_classes)
    
    train_data, train_labels = reformat(data["train_data"], data["train_labels"])
    valid_data, valid_labels = reformat(data["valid_data"], data["valid_labels"])
    test_data, test_labels = reformat(data["test_data"], data["test_labels"])

    return (
        (train_data, train_labels),
        (valid_data, valid_labels),
        (test_data, test_labels),
    )

def load_hsk_data_as_binary_label(num_classes, target_class):
    """
    Input
    -----
    target_class: Int - the int value corresponding to a single target class
    """
    data = open_pickle(num_classes)

    train_data = reformat_img_data(data["train_data"])
    train_labels = reformat_to_binary_labels(data["train_labels"], target_class=target_class)

    valid_data = reformat_img_data(data["valid_data"])
    valid_labels = reformat_to_binary_labels(data["valid_labels"], target_class=target_class)

    test_data = reformat_img_data(data["test_data"])
    test_labels = reformat_to_binary_labels(data["test_labels"], target_class=target_class)

    return (
        (train_data, train_labels),
        (valid_data, valid_labels),
        (test_data, test_labels),
    )

def main():
    # Generate pickles for 100 classes out of all HSK levels
    # bmps_to_pickle()

    # Generate pickles for 25 classes out HSK 1
    # bmps_to_pickle(num_classes=25, hsk_levels=(1,))

    # Generate pickles for 10 classes out HSK 1
    # bmps_to_pickle(num_classes=10, hsk_levels=(1,))
    pass

if __name__=="__main__":
    main()
