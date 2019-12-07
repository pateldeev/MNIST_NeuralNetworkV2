import cv2
import data_loader as dl
import network
import numpy as np

base_data_directory = "/home/dp/PycharmProjects/NeuralNetwork1/data/"

if __name__ == "__main__":
    net = network.Network(network_size=[784, 16, 16, 10])

    # net.randomize_weights_and_biases()
    # net.save_weights_and_biases(base_data_directory)
    net.read_weights_and_biases(base_data_directory)

    # train_data = dl.get_images(base_data_directory + "train-images-idx3-ubyte")
    # dl.save_images(train_data, base_data_directory + "train/")
    train_data = dl.read_images(base_data_directory + "train/", read_limit=3)

    train_labels = dl.get_labels(base_data_directory + "train-labels-idx1-ubyte")

    net_in = [pixel / 255 for row in train_data[0] for pixel in row]
    print("Input to network: ", net_in)
    net_out = net.run_network(net_input=net_in)
    print("Output of network: ", net_out)

    total_cost = net.compute_cost([net_in], [train_labels[0]])
    print("COST: ", total_cost)

    print("Running back_prop")
    net.back_prop(net_in, train_labels[0])
    print("Done running back_prop")

    cv2.imshow("Window", np.array(train_data[0], dtype=np.uint8, ndmin=2))
    cv2.waitKey()

    print("Done!")
