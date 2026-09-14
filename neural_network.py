import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

functions = {
    "ReLU": lambda x: np.maximum(0, x),
    "Sigmoid": sigmoid,
    "Tanh": lambda x: np.tanh(x),
}

derivatives = {
    "ReLU": lambda x: np.where(x > 0, 1.0, 0.0),
    "Sigmoid": sigmoid_derivative,
    "Tanh": lambda x: 1 - np.tanh(x) ** 2,
}

class Layer:
    def __init__(self, size, input_size, activation_name):
        self.activation_func = functions[activation_name]
        self.activation_deriv_func = derivatives[activation_name]

        self.input_size = input_size
        self.size = size

        self.weights = np.ones((size, input_size), dtype=float)
        self.biases = np.zeros(size, dtype=float)

        self.activations = np.zeros(size, dtype=float)
        self.weighted_sums = np.zeros(size, dtype=float)
        self.inputs = np.zeros(input_size, dtype=float)

        self.weight_gradients = np.zeros((size, input_size), dtype=float)
        self.bias_gradients = np.zeros(size, dtype=float)
        self.weight_velocities = np.zeros((size, input_size), dtype=float)
        self.bias_velocities = np.zeros(size, dtype=float)

    def calculate_outputs(self, inputs):
        # inputs shape: (N, input_size) where N is total pixels (10000)
        # weights shape: (size, input_size)
        # np.dot(inputs, self.weights.T) gives shape (N, size)
        # self.biases matches across the rows automatically
        self.inputs = inputs
        self.weighted_sums = np.dot(inputs, self.weights.T) + self.biases
        self.activations = self.activation_func(self.weighted_sums)
        return self.activations

    def randomize(self):
        # FOR RELU
        self.weights = np.random.randn(self.size, self.input_size) * np.sqrt(2 / self.input_size)
        self.biases = np.zeros(self.size)

    def apply_gradients(self, learn_rate, momentum):
        self.bias_velocities = momentum * self.bias_velocities - learn_rate * self.bias_gradients
        self.weight_velocities = momentum * self.weight_velocities - learn_rate * self.weight_gradients

        self.biases += self.bias_velocities
        self.weights += self.weight_velocities

    def clear_gradients(self):
        self.bias_gradients[...] = 0
        self.weight_gradients[...] = 0

    @staticmethod
    def node_cost_derivative(output, expected):
        return 2 * (output - expected)

    def get_output_derivatives(self, expected):
        # calculates the partial derivative of the output layer's weighted sum w/ respect to the cost
        # z --> a --> c (a=activation/output; z=weighted_sum; c=cost)
        # ∂c/∂z = ∂a/∂z * ∂c/∂a (this is the output)
        # ∂c/∂z = A'(z) * C'(a)
        activation_derivatives = self.activation_deriv_func(self.weighted_sums)
        cost_derivatives = self.node_cost_derivative(self.activations, expected)
        node_derivatives = activation_derivatives * cost_derivatives
        return node_derivatives # so this is actually the weighted sum derivative (NOT the activation)

    def get_hidden_derivatives(self, last_layer, last_node_derivatives):
        # calculates the partial derivative of the hidden layer's weighted sum w/ respect to the old partial derivative
        # z1 --> a1 --> z --> a --> c (a=activation/output; z=weighted_sum; c=cost)
        #          \--> z --> a ----^
        # ∂c/∂z1 = ∂a1/∂z1 * ∂z/∂a1 * (∂a/∂z * ∂c/∂a) (this is the output, it reuses the previous partial derivative)
        # ∂c/∂z1 = A'(z1) * weight * old_deriv
        # a1 actually has many connections since z = b + a1*w1 + a2*w2 + ...
        # we can use the multivariable chain rule (just add up all the partial derivatives that affects the cost)
        # ∂c/∂z1 = A'(z1) * sum(weights * old_derivatives)
        activation_derivatives = self.activation_deriv_func(self.weighted_sums)
        weighted_sum_derivatives = np.dot(last_node_derivatives, last_layer.weights)
        node_derivatives = activation_derivatives * weighted_sum_derivatives
        return node_derivatives

    def update_gradients(self, node_derivatives):
        # (a & w & b) --> z --> node_derivatives (∂c/∂z)
        # the gradient of 'z' is affected by w and b
        # z = sum(a*w)+b
        # ∂z/∂w = a (a directly affects the gradient)
        # ∂z/∂b = 1 (b only shifts the graph up, so the gradient stays the same)

        # outer multiplies all combination of pairs into a 2d matrix
        self.weight_gradients += node_derivatives.T @ self.inputs # @ is the same as dot product REMEMBER
        self.bias_gradients += np.sum(node_derivatives, axis=0)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation="ReLU", output_activation="Sigmoid"):
        self.layers = []
        layer_amount = len(layer_sizes)
        for i in range(1, layer_amount):
            layer = Layer(layer_sizes[i], layer_sizes[i - 1], activation if i < layer_amount - 1 else output_activation)
            self.layers.append(layer)

    def calculate_outputs(self, activations):
        for layer in self.layers:
            activations = layer.calculate_outputs(activations)
        return activations

    def accuracy(self, inputs, expected):
        predicted = np.argmax(self.calculate_outputs(inputs), axis=1)
        if expected.ndim == 2:
            expected = np.argmax(expected, axis=1)
        return np.average(predicted == expected)

    def randomize(self):
        for layer in self.layers:
            layer.randomize()

    def cost(self, inputs, expected) -> float:
        outputs = self.calculate_outputs(inputs)
        node_costs = np.sum(np.power(expected - outputs, 2), axis=1)
        return np.average(node_costs)

    def apply_gradients(self, learn_rate, momentum):
        for layer in self.layers:
            layer.apply_gradients(learn_rate, momentum)

    def backward(self, inputs, expected):
        # Using back propagation to find the gradients of all weights and biases
        self.calculate_outputs(inputs) # this saves the weighted inputs and activations (to use for backprop)

        # update gradients of output layer
        output_layer = self.layers[-1]
        node_derivatives = output_layer.get_output_derivatives(expected)
        output_layer.update_gradients(node_derivatives)

        # update for hidden layers
        for i in range(len(self.layers)-2, -1, -1):
            node_derivatives = self.layers[i].get_hidden_derivatives(self.layers[i+1], node_derivatives)
            self.layers[i].update_gradients(node_derivatives)

    def clear_gradients(self):
        for layer in self.layers:
            layer.clear_gradients()

    def learn(self, inputs, expected, learn_rate, momentum=0.9):
        # Using derivatives and chain rule to calculate the gradient with back propagation
        # clear first cuz we'll be adding the gradients up
        self.clear_gradients()

        # for each backwards step we add it to the gradient
        self.backward(inputs, expected)

        # because the current "gradient" is actually the sum of all calculated gradients
        # we should divide by the batch size to get the average
        self.apply_gradients(learn_rate/len(inputs), momentum)

    def slow_learn(self, inputs, expected, learn_rate):
        # Works by changing all weights and biases by h in order to approximate the gradient (f(x+h)-f(x))/h
        # Though this is slow since we have to calculate the cost for every weight and bias

        start_cost = self.cost(inputs, expected)
        h = 0.001

        for layer in self.layers:

            for i in range(layer.size):
                layer.biases[i] += h
                delta_cost = self.cost(inputs, expected) - start_cost
                layer.biases[i] -= h
                layer.bias_gradients[i] = delta_cost/h

                for j in range(layer.input_size):
                    layer.weights[i][j] += h
                    delta_cost = self.cost(inputs, expected) - start_cost
                    layer.weights[i][j] -= h
                    layer.weight_gradients[i][j] = delta_cost/h

        self.apply_gradients(learn_rate, 0)