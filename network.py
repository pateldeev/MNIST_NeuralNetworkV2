import math
import numpy as np
import pickle


# Vectorized sigmoid function.
def sigmoid_vectorized(v):
    return 1 / (1 + np.exp(-v))


# Derivative of sigmoid function. Expects sigmoid_vectorized to already be called on input vector.
# Basically implements sigam' = sigma*(1-sigma) over all the values of a vector
def sigmoid_prime_vectorized(v_sigmoid):
    return list(map(lambda v_i: v_i * (1 - v_i), v_sigmoid))


# Compute expected value from label.
def compute_expected(label):
    assert label in range(10), "A label must be an integer in [0,9]."
    return [1.0 if i == label else 0.0 for i in range(10)]


class Network:
    def __init__(self, network_size):
        assert len(network_size) > 1, "Network must have at least 2 layers!"
        self.size = network_size

        # Allocate space to hold activation values.
        self.activations = [np.empty(shape=l_size, dtype=float) for l_size in network_size]

        # Allocate space to hold weights.
        self.weights = [np.empty(shape=(network_size[l_num], network_size[l_num - 1]), dtype=float)
                        for l_num in range(1, len(network_size))]

        # Allocate space to hold biases.
        self.biases = [np.empty(shape=l_size, dtype=float) for l_size in network_size[1:]]

    # Randomly assign weights and biases in [0, 1].
    def randomize_weights_and_biases(self):
        for l_num, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.weights[l_num] = np.random.rand(*w.shape)
            self.biases[l_num] = np.random.rand(*b.shape)

    # Save weights and biases to file.
    def save_weights_and_biases(self, file_path):
        with open(file_path, 'wb') as output_file:
            for w, b in zip(self.weights, self.biases):
                w_dump, b_dump = w.dumps(), b.dumps()
                output_file.write(len(w_dump).to_bytes(4, byteorder="big"))
                output_file.write(w_dump)
                output_file.write(len(b_dump).to_bytes(4, byteorder="big"))
                output_file.write(b_dump)

    # Read weights and biases from file.
    def read_weights_and_biases(self, file_path):
        with open(file_path, 'rb') as input_file:
            for l_num in range(len(self.size) - 1):
                temp = int.from_bytes(input_file.read(4), byteorder="big")
                self.weights[l_num] = pickle.loads(input_file.read(temp))
                temp = int.from_bytes(input_file.read(4), byteorder="big")
                self.biases[l_num] = pickle.loads(input_file.read(temp))

    # Feed input forward through network.
    # Input must a be list of the right size with values in the range [0, 1].
    # Sets the activation values to according the to input.
    def feed_forward(self, net_input):
        assert len(net_input) == self.size[0], "Network input must be the same size as the first layer of the network"
        assert all(0.0 <= i <= 1.0 for i in net_input), "All values in the Network input must be in range [0,1]"

        # Activation values of first layer is just the input.
        self.activations[0] = np.array(net_input, dtype=float)

        # Compute activation values of each successive layer in the network.
        for l_num, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.activations[l_num + 1] = sigmoid_vectorized((w @ self.activations[l_num]) + b)

    # Run network on input.
    # Input must a be list of the right size with values in the range [0, 1].
    # Returns the predicted value of the network.
    def run_network(self, net_input):
        # Feed forward input.
        self.feed_forward(net_input)

        # Return the predicted value of the network.
        return int(np.argmax(self.activations[-1]))

    # Computes cost of network over a list of inputs.
    # The labels represent the integer value of the inputs.
    def compute_cost(self, net_inputs, labels):
        assert len(net_inputs) == len(labels), "Labels and inputs must be of the same size!"

        total_cost = 0.0
        for net_input, label in zip(net_inputs, labels):
            # Compute expected values of final layer.
            y = compute_expected(label)

            # Feed forward input through network.
            self.feed_forward(net_input)

            # Compute cost using the final layers activation values.
            for y_actual, y_expected in zip(self.activations[-1], y):
                total_cost += math.pow(y_actual - y_expected, 2)

        return total_cost

    def back_prop(self, training_imgs, training_labels):
        assert len(training_imgs) == len(training_labels), "Each training image must have a label!"

        # Allocate space to hold all the necessary derivatives.
        # Derivative of cost with respect to activations
        dc_da = [np.empty(shape=l_size, dtype=float) for l_size in self.size]
        # Derivative of cost with respect to weights.
        dc_db = [np.empty(shape=l_size, dtype=float) for l_size in self.size[1:]]
        # Derivative of cost with respect to biases.
        dc_dw = [np.empty(shape=(self.size[l_num], self.size[l_num - 1]), dtype=float)
                 for l_num in range(1, len(self.size))]

        # Iterate over all the training images and labels
        for img, label in zip(training_imgs, training_labels):
            # Feed forward image to compute activation values.
            self.feed_forward(img)

            # Compute expected values of final layer.
            y = compute_expected(label)

            # Compute the sigma'(activations). Needed for efficiency.
            d_sigmoid_activations = [np.array(sigmoid_prime_vectorized(a)) for a in self.activations]

            # Compute dc_da of final layer using cost function directly.
            dc_da[-1] = np.array([2 * (a_f_i - y_i) for a_f_i, y_i in zip(self.activations[-1], y)])

            # Use the dc_da for the final layer to back propagate through network.
            for l_num in range(len(self.size) - 1, 0, -1):
                # Compute derivative of cost with respect to biases in layer.
                dc_db[l_num - 1] = dc_da[l_num] * d_sigmoid_activations[l_num]

                # Compute derivative of cost with respect to weights in layer.
                for r, dc_db_r in enumerate(dc_db[l_num - 1]):
                    dc_dw[l_num - 1][r] = dc_db_r * self.activations[l_num - 1]

                # Compute derivative of cost with respect to activations in layer. Needed for next iteration.
                for j in range(dc_da[l_num - 1].size):
                    dc_da[l_num - 1][j] = np.dot(dc_db[l_num - 1], self.weights[l_num - 1].transpose()[j])

            # Move weights and biases in direction of negative gradient.
            for l_num, (dw, db) in enumerate(zip(dc_dw, dc_db)):
                self.weights[l_num] -= np.divide(dw, len(training_imgs))
                self.biases[l_num] -= np.divide(db, len(training_imgs))
