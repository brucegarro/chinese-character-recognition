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
