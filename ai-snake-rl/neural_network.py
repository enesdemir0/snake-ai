import numpy as np
from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=-1, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def activation_function(x):
    return 1 / (1 + np.exp(-x))


def activation_derivative(x):
    sig = activation_function(x)
    return sig * (1 - sig)


class NeuralNetwork:
    def __init__(self, no_of_in_nodes, no_of_hidden_nodes, no_of_out_nodes, learning_rate, bias=1):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        bias_node = 1 if self.bias else 0
        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes + bias_node))

        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes + bias_node))

    def train(self, input_vector, target_vector):
        input_vector = np.array(input_vector).reshape(-1, 1)
        if self.bias:
            input_vector = np.vstack([input_vector, [[self.bias]]])
        target_vector = np.array(target_vector).reshape(-1, 1)

        hidden_input = self.weights_in_hidden @ input_vector
        hidden_output = activation_function(hidden_input)

        if self.bias:
            hidden_output = np.vstack([hidden_output, [[self.bias]]])

        final_input = self.weights_hidden_out @ hidden_output
        final_output = final_input

        output_error = target_vector - final_output
        delta_output = output_error

        self.weights_hidden_out += self.learning_rate * (delta_output @ hidden_output.T)

        hidden_error = self.weights_hidden_out.T @ delta_output
        hidden_error_no_bias = hidden_error[:-1, :] if self.bias else hidden_error
        delta_hidden = hidden_error_no_bias * activation_derivative(hidden_input)

        self.weights_in_hidden += self.learning_rate * (delta_hidden @ input_vector.T)

    def run(self, input_vector):
        input_vector = np.array(input_vector).reshape(-1, 1)
        if self.bias:
            input_vector = np.vstack([input_vector, [[self.bias]]])

        hidden_input = self.weights_in_hidden @ input_vector
        hidden_output = activation_function(hidden_input)

        if self.bias:
            hidden_output = np.vstack([hidden_output, [[self.bias]]])

        final_input = self.weights_hidden_out @ hidden_output
        final_output = final_input
        return final_output

    def save(self, filename):
        np.savez(filename,
                 weights_in_hidden=self.weights_in_hidden,
                 weights_hidden_out=self.weights_hidden_out)

    def load(self, filename):
        data = np.load(filename)
        self.weights_in_hidden = data['weights_in_hidden']
        self.weights_hidden_out = data['weights_hidden_out']
