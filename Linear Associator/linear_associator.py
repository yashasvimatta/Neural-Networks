# Matta, Yashasvi
# 1002_091_131
# 2022_10_09
# Assignment_02_01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        self.number_of_nodes = number_of_nodes
        self.input_dimensions = input_dimensions
        self.transfer_function = transfer_function
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions)
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """

        self.initialize_weights()

    def initialize_weights(self, seed=None):
        if seed != None:
            np.random.seed(seed)
            self.weights =np.random.randn(self.number_of_nodes,self.input_dimensions)
            self.set_weights(self.weights)
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """

    def set_weights(self, W):
        self.weights = W
        if W.shape == (self.number_of_nodes,self.input_dimensions):
            return None
        else:
            return -1
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """

    def get_weights(self):
        return self.weights
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """

    def predict(self, X):
        # as the shape is same so now we can multiply 
        net = np.dot(self.weights,X)
        # now passing through hard limit activation function
        if self.transfer_function == "Hard_limit":
            for x in net:
                for i in range(0,len(x)):
                    if x[i] >= 0:
                        x[i] = 1
                    else:
                        x[i] = 0
        return net
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """

    def fit_pseudo_inverse(self, X, y):

        self.weights = np.dot(y,np.linalg.pinv(X))
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        for epochs in range(num_epochs):
            for i in range(0,X.shape[1]):
                current_input= X[:,i].reshape(self.input_dimensions,-1)
                current_net = self.predict(current_input)
                target_output = y[:,i].reshape(self.number_of_nodes,-1)
                if learning.casefold() == "delta":
                    self.weights = self.weights + (np.dot(np.subtract(target_output,current_net), current_input.T))*alpha
                elif learning.casefold() == "filtered":
                    self.weights = (1-gamma) * self.weights + np.dot(target_output,current_input.T) *alpha
                else:
                    self.weights = self.weights + np.dot(current_net,current_input.T) * alpha
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """

    def calculate_mean_squared_error(self, X, y):
        mse = (np.square(np.subtract(y,self.predict(X)))).mean()
        return mse
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """