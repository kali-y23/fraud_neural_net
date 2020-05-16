import numpy as np
import matplotlib.pyplot as plt

"""
Deep Neural Net, which uses tanh activation function in the hidden layers and sigmoid in the output layer.
Cross-entropy is used as a loss function
Intended to classify observations as 1 or 0
"""

class DeepNeuralNet:
    def __init__(self, dimensions, X_train, Y_train, learning_rate=0.0075, num_iterations=2500, print_cost=False):
        """
        :param dimensions: python array (list) containing the dimensions of each layer in the network,
                           i.e. size of the input layer, all hidden layers, output layer
        :param X_train: numpy array of size (n, m), where n = number of observations and m = number of variables
        :param Y_train: numpy array of size (n, 1), where n = number of observations
        :param learning_rate: learning rate of the model, used during the parameters update
        :param num_iterations: number of iterations for model training
        :param print_cost: if True, current value of the loss function will be printed every 100 iterations during the training
        """

        self.dimensions = dimensions
        self.X = X_train
        self.Y = Y_train
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost

    @staticmethod
    def tanh(x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))

    def init_parameters(self):
        np.random.seed(765976208)
        parameters = {}
        layer_dims = self.dimensions
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def forward_propagation_step(self, A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        assert (Z.shape == (W.shape[0], A_prev.shape[1]))

        if activation == "tanh":
            A = self.tanh(Z)
        elif activation == "sigmoid":
            A = self.sigmoid(Z)
        else:
            raise Exception("Invalid activation (forward propagation)")

        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        linear_cache = (A_prev, W, b)
        activation_cache = Z

        return A, (linear_cache, activation_cache)


    def forward_propagation(self, X, parameters):
        """
        tanh * (L-1) -> sigmoid computation

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                  every cache of forward_propagation_step() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        for l in range(1, L):
            A_prev = A
            A, cache = self.forward_propagation_step(A, parameters["W" + str(l)], parameters["b" + str(l)], "tanh")
            caches.append(cache)

        Y_hat, cache = self.forward_propagation_step(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
        caches.append(cache)

        assert (Y_hat.shape == (1, X.shape[1]))

        return Y_hat, caches


    def compute_cost(self, Y_hat):
        Y = self.Y

        m = Y.shape[1]  # number of train set observations

        cross_entropy = (Y * np.log(Y_hat)) + ((1 - Y) * (np.log(1 - Y_hat)))
        cost = (-1 / m) * np.sum(cross_entropy)

        cost = np.squeeze(cost)  # To make sure your cost's shape is a number, e.g. this turns [[17]] into 17.
        assert (cost.shape == ())

        return cost

    def back_propagation_step(self, dA, cache, activation):
        """
        :param Y_hat: output of the neural net from the last forward propagation
        :param cache: tuple containing cache from the last forward propagation step: ((A_prev, W, b), Z)
        :param activation: "sigmoid" or "tanh"
        :return: gradients of A(ctivation), W and b
        """

        linear_cache, Z = cache

        if activation == "tanh":
            A = self.tanh(Z)
            dZ = dA * (1 - np.power(A, 2))

        elif activation == "sigmoid":
            A = self.sigmoid(Z)
            dZ = dA * (1 - A) * A
        else:
            raise Exception("Invalid activation (back propagation)")

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        assert (dA_prev.shape == A_prev.shape)

        return dA_prev, dW, db

    def back_propagation(self, Y_hat, caches):
        """
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """

        grads = {}
        L = len(caches)  # the number of layers

        Y = self.Y
        Y = Y.reshape(Y_hat.shape)
        dA = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.back_propagation_step(dA, current_cache, "sigmoid")

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.back_propagation_step(grads["dA" + str(l + 1)], current_cache, "tanh")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2  # number of layers in the neural network

        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.learning_rate * grads["db" + str(l + 1)]

        return parameters

    def train(self):
        parameters = self.init_parameters()
        costs = []

        # Gradient descent
        for i in range(0, self.num_iterations):
            Y_hat, caches = self.forward_propagation(self.X, parameters)

            cost = self.compute_cost(Y_hat)

            grads = self.back_propagation(Y_hat, caches)

            parameters = self.update_parameters(parameters, grads)

            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 100 == 0:
                costs.append(cost)

        return costs, parameters

    def plot_costs(self, costs):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()


    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.

        :param X -- data set of examples you would like to label
        :param parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        predictions = np.zeros((1, m))

        # Forward propagation
        probabilities, caches = self.forward_propagation(X, parameters)

        for i in range(0, probabilities.shape[1]):
            if probabilities[0, i] > 0.5:
                predictions[0, i] = 1
            else:
                predictions[0, i] = 0

        print("Accuracy: " + str(np.sum((predictions == y) / m)))

        return predictions
