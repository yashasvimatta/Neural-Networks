# Matta, Yashasvi
# 1002_091_131
# 2022_09_25
# Assignment_01_01

import numpy as np

class SingleLayerNN(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        self.number_of_nodes = number_of_nodes
        self.input_dimensions = input_dimensions
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions+1)
        self.bias =  np.random.randn(self.number_of_nodes,1)
        self.initialize_weights()
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
    
    def initialize_weights(self,seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
            self.weights =np.random.randn(self.number_of_nodes,self.input_dimensions+1)
            self.set_weights(self.weights)
            

    def set_weights(self, W):
        self.weights = W
        if W.shape == (self.number_of_nodes,self.input_dimensions+1):
            return None
        else:
            return -1

        """
        This function sets the weight matrix (Bias is included in the weight matrix).
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
        X = np.concatenate((np.ones((1, X.shape[1]),dtype=int),X), axis=0)
        # as the shape is same so now we can multiply 
        net = np.dot(self.weights,X)
        # now passing through hard limit activation function
        for x in net:
            for i in range(0,len(x)):
                if x[i] >= 0:
                    x[i] = 1
                else:
                    x[i] = 0
        return net
        """
        Make a prediction on a batch of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """

    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        for epochs in range(num_epochs):
            for i in range(0,X.shape[1]):
                current_input= X[:,i].reshape(self.input_dimensions,-1)
                current_net = self.predict(current_input)
                target_output = Y[:,i].reshape(self.number_of_nodes,-1)
                error = np.subtract(target_output,current_net)
                current_input = np.concatenate((np.ones((1,1),dtype=int),current_input), axis=0)
                self.weights = self.weights + (np.dot(error, current_input.T))*alpha
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """

    def calculate_percent_error(self,X, Y):
        error_counter = 0
        for i in range(0,X.shape[1]):
            current_input= X[:,i].reshape(self.input_dimensions,-1)
            current_net = self.predict(current_input)
            target_output = Y[:,i].reshape(self.number_of_nodes,-1)
            error = np.subtract(target_output,current_net)
            if np.any(error):
               error_counter += 1
        percent_error = 100*(error_counter / X.shape[1])
        return percent_error
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """

if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2
    print()

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())