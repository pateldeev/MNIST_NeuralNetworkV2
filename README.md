# MNIST_NeuralNetworkV2
Simple neural network to classify and written digits. Impliments the back propagation algorithm.
Version 2 of https://github.com/pateldeev/MNIST_NeuralNetwork

# Data Set 
http://yann.lecun.com/exdb/mnist/

# Credit/Inspiration
https://youtu.be/aircAruvnKk

# Running
## Requirements
Requires python3. (Tested with [3.7.3](https://www.python.org/downloads/release/python-373/)).
Make sure to have [opencv](https://pypi.org/project/opencv-python/) and [numpy](https://numpy.org/)
 * `pip install opencv-python-headless`
 * `pip install numpy`
## Decompressing Data
This repository contains all the images in two compressed files ([one](https://github.com/pateldeev/MNIST_NeuralNetworkV2/blob/master/data/train-images-idx3-ubyte) with 60000 training images) and ([one](https://github.com/pateldeev/MNIST_NeuralNetworkV2/blob/master/data/t10k-images-idx3-ubyte) with 10000 validation images). The repo also contains two files with the labels for these images. They should be decompressed and saved as .png files. The [data_loader](https://github.com/pateldeev/MNIST_NeuralNetworkV2/blob/master/data_loader.py) module can accomplish this. Simply uncomment the appropriate code in `main.py`
