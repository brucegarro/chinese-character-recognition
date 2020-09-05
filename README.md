# Chinese Character Recognition
Neural networks to idenity Chinese characters from photographs of handwritten samples.

## Getting started

The following covers how to train a convolutional neural network which accurately classifies between 50 types of Chinese characters. The instructions include information on obtaining and cleaning the data set as well as training models locally via GPU on an Ubuntu machine.

## Prerequisites

The training code was developed on a single instance Ubuntu 18.04.1 machine using an Nvidia GeForce RTX 2070 8GB GPU. The code hasn't been tested using other operating systems or CPUs. The code runs best on a system with at least 32GB of memory for data preprocessing. The code is implemented in Python 2.7.

This project uses Tensorflow and CUDA to perform parallel processing on Nvidia GPUs. I've run this project successfully with a manually installed version of CUDA 10.0. The following resource may be useful to configure your machine to run CUDA.
https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070

## Download Data Sets

The CASIA Online and Offline Dataset is made available by NLPR and is free for academic use.
http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

Data sets used for this project are: Gnt1.0Train {part1,part2,part3}, Gnt1.0Test, Gnt1.1Train {part1,part2}, Gnt1.1Test, Gnt1.2Train {part1,part2}, Gnt1.2Test, competition-gnt (Offline Character Data). You can successfully train models for fewer classes on a subset of the above data sets however, I had best results by using all of the training data sets listed above at the same time.

* Download the data sets from the link above and unzip each `.zip` file into a its own folder per zip file.

### Data Format

The raw data for each image is stored in a struct format stored in `.gnt` files. The cleaning code detailed later in these instructions converts the `.gnt` files to `.bmp` image files. These `.bmp` files read into Numpy arrays and then fed into the model.

## Installation and Use

1. Configure settings and DATA_PATH

* Copy `settings/local.py.example` to `settings/local.py` and define the `DATA_PATH` to your data folder.
```
cd ~/{PROJECT ROOT FOLDER}
cp settings/local.py.example settings/local.py
```
* In `local.py` set the `DATA_PATH` to the location where you have downloaded the CASIA datasets

2. Create a python virtual environment and install requirements.txt
```
cd ~/{YOUR VIRTUAL ENVIRONMENTS FOLDER}/ccr
virtualenv ccr
source ~/{YOUR VIRTUAL ENVIRONMENTS FOLDER}/ccr/bin/activate

cd ~/{PROJECT ROOT FOLDER}
pip install -r requirements.txt
```

3. Clean Data

Coming Soon

4. Run 50 Character Model

Coming Soon

# Acknowledgments

This project is motivated by research conducted by Yuhao Zhang in their paper "Deep Convolutional Network for Handwritten Chinese Character Recognition"
yuhao.im/files/Zhang_CNNChar.pdf
