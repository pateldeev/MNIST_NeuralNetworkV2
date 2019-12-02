import math
import numpy as np
import pickle


# Vectorized sigmoid function.
def sigmoid_vectorized(v):
    return 1 / (1 + np.exp(-v))


# Derivative of sigmoid function.
def sigmoid_prime(x):
    x_sigmoid = sigmoid_vectorized(x)
    return x_sigmoid * (1 - x_sigmoid)


class Network:
    def __init__(self, network_size):
        self.size = network_size

        # Allocate space to hold weights and biases.
        self.weights = [np.empty(shape=(network_size[layer_num], network_size[layer_num - 1]), dtype=float)
                        for layer_num in range(1, len(network_size))]
        self.biases = [np.empty(shape=layer_size, dtype=float) for layer_size in network_size[1:]]

    # Randomly assigned weights and biases [0, 1].
    def randomize_weights_and_biases(self):
        for layer_num, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            self.weights[layer_num] = np.random.rand(*weights.shape)
            self.biases[layer_num] = np.random.rand(*biases.shape)

    # Save weights and biases to file.
    def save_weights_and_biases(self, base_dir, file_name="network_params.byte"):
        with open(base_dir + file_name, 'wb') as output_file:
            for weights, biases in zip(self.weights, self.biases):
                weights_dump, biases_dump = weights.dumps(), biases.dumps()
                output_file.write(len(weights_dump).to_bytes(4, byteorder="big"))
                output_file.write(weights_dump)
                output_file.write(len(biases_dump).to_bytes(4, byteorder="big"))
                output_file.write(biases_dump)

    # Read weights and biases from file.
    def read_weights_and_biases(self, base_dir, file_name="network_params.byte"):
        with open(base_dir + file_name, 'rb') as input_file:
            for layer_num in range(len(self.size) - 1):
                temp = int.from_bytes(input_file.read(4), byteorder="big")
                self.weights[layer_num] = pickle.loads(input_file.read(temp))
                temp = int.from_bytes(input_file.read(4), byteorder="big")
                self.biases[layer_num] = pickle.loads(input_file.read(temp))

    # Run network on input.
    # net_input must a be list of the right size with values in the range [0, 1].
    # returns output after running network. A list of values in the range [0, 1].
    def run_network(self, net_input):
        if len(net_input) != self.size[0]:
            raise ValueError("Network input has incorrect dimensions!")

        # Activation values of first layer is just the input.
        activations = np.array(net_input, dtype=float)

        # Compute activation values of each successive layer in the network.
        for weights, biases in zip(self.weights, self.biases):
            activations = sigmoid_vectorized(np.matmul(weights, activations) + biases)

        return activations.tolist()

    # Computes cost of network over a list of inputs.
    # The labels represent the integer value of the inputs.
    def compute_cost(self, net_inputs, labels):
        if len(net_inputs) != len(labels):
            raise ValueError("Labels and inputs must be of the same size!")

        total_cost = 0.0
        for net_input, label in zip(net_inputs, labels):
            net_output = self.run_network(net_input)
            y = [1.0 if i == label else 0.0 for i in range(10)]
            print("Y: ", y)
            for y_actual, y_expected in zip(net_output, y):
                total_cost += math.pow(y_actual - y_expected, 2)

        return total_cost
