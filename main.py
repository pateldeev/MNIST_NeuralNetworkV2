import cv2
import data_loader as dl
import network
import numpy as np
import os 

BASE_DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/"
NET_PARAMS_FILE = BASE_DATA_DIR + "network_params.byte"

TRAIN_DATA_FILE = BASE_DATA_DIR + "train-images-idx3-ubyte"
TRAIN_LABEL_FILE = BASE_DATA_DIR + "train-labels-idx1-ubyte"
TRAIN_DATA_SIZE = 60000

TEST_DATA_FILE = BASE_DATA_DIR + "t10k-images-idx3-ubyte"
TEST_LABEL_FILE = BASE_DATA_DIR + "t10k-labels-idx1-ubyte"
TEST_DATA_SIZE = 10000

BASE_IMG_SAVE_DIR = BASE_DATA_DIR + "img/"

BATCH_SIZE = 100

if __name__ == "__main__":
    net = network.Network(network_size=[784, 16, 16, 10])

    net.randomize_weights_and_biases()
    net.save_weights_and_biases(NET_PARAMS_FILE)
    net.read_weights_and_biases(NET_PARAMS_FILE)

    # train_data = dl.get_images(TRAIN_DATA_FILE)
    # dl.save_images(train_data, BASE_IMG_SAVE_DIR)
    train_data = dl.read_images(BASE_IMG_SAVE_DIR, start_index=0, read_limit=BATCH_SIZE)
    
    train_labels = dl.get_labels(TRAIN_LABEL_FILE)

    # net_in = [pixel / 255 for row in train_data[0] for pixel in row]
    # print("Input to network:", net_in)
    # net_out = net.run_network(net_input=net_in)
    # print("Output of network:", net_out)
    # print("Activations:", net.activations[-1])

    for i in range(10):
        print("Running back_prop", i + 1)

        batch_imgs = [[pixel / 255 for row in img for pixel in row] for img in train_data[0:BATCH_SIZE]]
        batch_labels = train_labels[0:BATCH_SIZE]
        net.back_prop(batch_imgs, batch_labels)
        total_cost = net.compute_cost(batch_imgs, batch_labels)

        print("COST", i + 1, ":", total_cost)

    # cv2.imshow("Window", np.array(train_data[0], dtype=np.uint8, ndmin=2))
    # cv2.waitKey()

    print("Done!")
