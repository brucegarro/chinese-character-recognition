# Chinese Character Recognition
Neural networks which identify Chinese characters from photographs of handwritten samples.

## Getting started

The following guide covers how to train a convolutional neural network which accurately classifies 50 types of Chinese characters. The instructions include information on obtaining and cleaning the dataset as well as training models locally via GPU on an Ubuntu machine.

## Prerequisites

The training code was developed on a single instance Ubuntu 18.04.1 machine using an Nvidia GeForce RTX 2070 8GB GPU; it runs best on a system with at least 32GB of memory for data preprocessing. The code is implemented in Python 2.7.

This project uses Tensorflow and CUDA to perform parallel processing on Nvidia GPUs. I've run this project successfully with a manually installed version of CUDA 10.0. The following resource may be useful to configure your machine to run CUDA.
https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070

## Download Datasets

The CASIA Online and Offline Dataset is made available by NLPR and is free for academic use.
http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

Datasets used for this project are: Gnt1.0Train {part1,part2,part3}, Gnt1.0Test, Gnt1.1Train {part1,part2}, Gnt1.1Test, Gnt1.2Train {part1,part2}, Gnt1.2Test, competition-gnt (Offline Character Data). You can successfully train models for fewer classes on a subset of the above datasets; however, I had best results by using all of the training datasets listed above at the same time.

1. Download the datasets from the link above and unzip each `.zip` file's contents into its own folder.

### Data Format

The raw data for each image is formatted in a struct data structure stored in `.gnt` files. The cleaning code detailed later in these instructions converts the `.gnt` files to `.bmp` image files. At runtime, these `.bmp` files are read into Numpy arrays and then fed into the model.

## Installation

1. Copy `settings/local.py.example` to `settings/local.py`.

```
cd ~/{PROJECT ROOT FOLDER}
cp settings/local.py.example settings/local.py
```

2. In `settings/local.py` set the `DATA_PATH` to the location where you have downloaded and unzipped the CASIA datasets. Ensure that `GNT_SOURCE_NAMES` is matching the dataset folders on your filesystem by changing the variables in this file or the folder names on your local filesystem.

3. Create a python virtual environment and install requirements.txt

```
cd ~/{YOUR VIRTUAL ENVIRONMENTS FOLDER}/ccr
virtualenv ccr
source ~/{YOUR VIRTUAL ENVIRONMENTS FOLDER}/ccr/bin/activate

cd ~/{PROJECT ROOT FOLDER}
pip install -r requirements.txt
```

## Prepare Data

1. Convert `.gnt` files to `.bmp`

    + `python load/gnts_to_bmps.py`

2. Create a python pickle file which caches the class label-to-filepath mappings

    + `python load/build_dataset.py # This runs function get_or_create_path_label_pickle`

## Train Model

Train a 50-class Chinese character recognition model

   + `python modeling/multi_conv_model_50_classes.py`
 
If the program is running successfully, the validation results of each training epoch are printed on the command line:

<p align="center">
    <img src="/images/training-epochs.png" height="40%" width="40%" style="text-align:center">
</p>

## Acknowledgments

This project is motivated by research conducted by Yuhao Zhang in their paper "Deep Convolutional Network for Handwritten Chinese Character Recognition".
http://yuhao.im/files/Zhang_CNNChar.pdf
