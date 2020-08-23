"""
Produces folders with bmps generated of the HWDB1.0-1.2 Chinese character dataset of gnt files
"""

import settings

from load.utils import open_gnt_file, write_image
from load.os_utils import get_all_gnt_filepaths_in_folderpath


def write_gnt_to_bmps(gnt_filepath, gnt_source_path):
    """
    Input
    -----
    gnt_filepath: str - the innput .gnt filt to convert to bmp
    gnt_source_path: str - the filepath where an image will be written to.
    """
    images = open_gnt_file(gnt_filepath)

    # Save image to bmp file
    writer  = gnt_filepath.split("/")[-1].split(".")[0]
    for image, label_name in images:
        write_image(label_name, image, writer, gnt_source_path)

def write_all_gnts_in_source_to_bmps(gnt_source_path):
    gnt_filepaths = get_all_gnt_filepaths_in_folderpath(gnt_source_path)
    for i, gnt_path in enumerate(gnt_filepaths):
        write_gnt_to_bmps(gnt_path, gnt_source_path)

def write_all_gnts_to_bmps(gnt_source_paths=settings.GNT_SOURCE_PATHS):
    for path in gnt_source_paths:
        write_all_gnts_in_source_to_bmps(gnt_source_path=path)


if __name__ == "__main__":
    write_all_gnts_to_bmps()
