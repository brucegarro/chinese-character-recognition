import os
from os.path import join
import scipy.misc
import struct
import numpy as np

from load.constants import TARGET_IMG_SIZE

def write_image(label_name, image, writer, gnt_source_path):
    """
    Input
    -----
    label_name: str - The character, usually a Chinese character
    image: np.ndarray - the image
    writer: str - the author's alias (e.g. C001-f-f)
    gnt_source_path: str - the filepath where an image will be written to.
    """

    writer_path = join(gnt_source_path, writer)
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    output_bmp_path = join(writer_path, "%s.bmp" % label_name)
    # write image to bmp
    try:
        if not os.path.exists(output_bmp_path):
            scipy.misc.imsave(output_bmp_path, image)
            print "Wrote: %s" % output_bmp_path
        else:
            pass
            # print "Skipped: %s" % output_bmp_path
    except TypeError:
        print "Invalid: %s" % output_bmp_path

def open_gnt_file(gnt_filepath):
    """
    Returns
    -------
    images: list[ (images: ndarray, label: str) ]
    """
    images = []
    with open(gnt_filepath, "rb") as f:
        while True:
            packed_length = f.read(4)
            if packed_length == '' or packed_length == ' ' or packed_length == b'':
                break
            length = struct.unpack("<I", packed_length)[0]

            # Get character label info
            label_name = f.read(2)
            # The 2020 datasets are better decoded with gbk instead of gb2312
            label_name = label_name.decode("gbk")

            # Get image dimension info
            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]

            # Get  image data
            raw_bytes = f.read(height*width)
            bytez = struct.unpack("{}B".format(height*width), raw_bytes)

            # Convert image to array
            image = np.array(bytez).reshape(height, width)
            image = scipy.misc.imresize(image, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))

            images.append((image, label_name))
    return images

def open_image_as_array(filepath):
    with open(filepath, "rb") as f:
        img = scipy.misc.imread(f, flatten=True)

        # if normalize_fn:
        #     # If normalize is True, normalize pixel range to to range [-0.5, 0.5]
        #     for i in range(len(img)):
        #         for j in range(len(img[0])):
        #             img[i][j] = (img[i][j]/255.0) - 0.5

    return img

def reshape_raw_img_data(img_data, num_channels=1):
    """
    Format
    -------
        img_data: (N, IMG_SIZE, IMG_SIZE, num_channels)
    """
    _, x_size, y_size = img_data.shape

    args = (-1, x_size, y_size, num_channels)
    img_data = img_data.reshape(args).astype(np.float32)
    return img_data

def pad_img_data(img_data, padding=16, padding_color_value=255):
    # Add white padding to images
    white_value = padding_color_value
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

def get_class_label_map(class_labels):
    class_label_map = {label: i for i, label in enumerate(class_labels)}
    return class_label_map

def gaussian_normalize_img_data(img_data):
    mean, std = img_data.mean(), img_data.std()
    img_data = (img_data - mean) / std
    return img_data

def create_image_and_label_data_set(
    path_label_data,
    class_label_map,
    padding=16,
    padding_color_value=255,
):
    num_samples = len(path_label_data)
    img_size = TARGET_IMG_SIZE

    image_dataset = np.ndarray((num_samples, img_size, img_size), dtype=np.float32)
    labels_dataset = np.ndarray(num_samples, dtype=np.int32)

    for i, (image_path, label) in enumerate(path_label_data):
        img = open_image_as_array(str(image_path))
        np.copyto(image_dataset[i], img)
        labels_dataset[i] = class_label_map[label]

    image_dataset = reshape_raw_img_data(image_dataset)
    image_dataset = pad_img_data(image_dataset, padding=padding, padding_color_value=padding_color_value)
    image_dataset = gaussian_normalize_img_data(image_dataset)

    labels_dataset = reformat_labels(labels_dataset)
    return image_dataset, labels_dataset

def train_valid_split(image_dataset, labels_dataset, train_frac=0.8, random_seed=0):
    num_samples = len(image_dataset)
    samples_idx = list(np.arange(num_samples))

    # Randomize
    np.random.seed(random_seed)
    np.random.shuffle(samples_idx)

    cutoff_idx = int(num_samples * train_frac)

    train_image_dataset = image_dataset[samples_idx[:cutoff_idx]]
    train_label_dataset = labels_dataset[samples_idx[:cutoff_idx]]
    valid_image_dataset = image_dataset[samples_idx[cutoff_idx:]]
    valid_label_dataset = labels_dataset[samples_idx[cutoff_idx:]]
    return train_image_dataset, train_label_dataset, valid_image_dataset, valid_label_dataset