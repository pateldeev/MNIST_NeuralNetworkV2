import cv2
import data_loader as dl
import network
import numpy as np
import os
from random import randint

BASE_DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/"
NET_PARAMS_FILE_RANDOM = BASE_DATA_DIR + "network_params_random.byte"
NET_PARAMS_FILE_TRAINED = BASE_DATA_DIR + "network_params_trained.byte"

TRAIN_DATA_FILE = BASE_DATA_DIR + "train-images-idx3-ubyte"
TRAIN_LABEL_FILE = BASE_DATA_DIR + "train-labels-idx1-ubyte"
TRAIN_DATA_SIZE = 60000

TEST_DATA_FILE = BASE_DATA_DIR + "t10k-images-idx3-ubyte"
TEST_LABEL_FILE = BASE_DATA_DIR + "t10k-labels-idx1-ubyte"
TEST_DATA_SIZE = 10000

BASE_IMG_SAVE_DIR = BASE_DATA_DIR + "img/"

BATCH_SIZE = 100


def randomize_and_save_network_parameters(n):
    n.randomize_weights_and_biases()
    n.save_weights_and_biases(NET_PARAMS_FILE_RANDOM)


def unpack_train_data():
    dl.save_images(dl.get_images(TRAIN_DATA_FILE), BASE_IMG_SAVE_DIR)


def get_train_labels():
    return dl.get_labels(TRAIN_LABEL_FILE)


def unpack_test_data():
    dl.save_images(dl.get_images(TEST_DATA_FILE), BASE_IMG_SAVE_DIR)


def get_test_labels(file_name):
    return dl.get_labels(TEST_LABEL_FILE)


if __name__ == "__main__":
    net = network.Network(network_size=[784, 16, 16, 10])

    # Load randomized weights and biases.
    # randomize_and_save_network_parameters(net)
    net.read_weights_and_biases(NET_PARAMS_FILE_RANDOM)

    # Unpack the train data.
    # unpack_train_data()

    train_labels = get_train_labels()

    start_index = 0
    for i in range(3000):
        print("Running back_prop", i + 1)

        # Load a batch of images and associate labels.
        data = dl.read_images(BASE_IMG_SAVE_DIR, start_index=start_index, read_limit=BATCH_SIZE)
        batch_imgs = [[pixel / 255 for row in img for pixel in row] for img in data]
        batch_labels = train_labels[start_index:start_index + BATCH_SIZE]

        # Perform back propagation on images
        net.back_prop(batch_imgs, batch_labels)

        # Print out cost on images in batch.
        total_cost = net.compute_cost(batch_imgs, batch_labels)
        print("COST ( ", start_index, "to", start_index + BATCH_SIZE - 1, "):", total_cost)

        # Increment index for next batch.
        start_index += BATCH_SIZE
        start_index %= TRAIN_DATA_SIZE

    # Save trained weights and biases.
    net.save_weights_and_biases(NET_PARAMS_FILE_TRAINED)

    # Test network output on random training values
    for _ in range(5):
        index = randint(0, TRAIN_DATA_SIZE - 1)
        img = dl.read_images(BASE_IMG_SAVE_DIR, start_index=index, read_limit=1)[0]

        net_input = [pixel / 255 for row in img for pixel in row]
        net_prediction = net.run_network(net_input)
        print(net.activations[-1])

        print("Network Prediction: ", net_prediction)

        cv2.imshow("Window", np.array(img, dtype=np.uint8, ndmin=2))
        cv2.waitKey()

    print("Done!")
