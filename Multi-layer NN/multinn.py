# Matta, Yashasvi
# 1002_091_131
# 2020_10_30
# Assignment_03_01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.transfer_function = []
        self.weights = []
        self.biases = []
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """

        self.num_nodes_list = []
    def add_layer(self, num_nodes, transfer_function="Linear"):
      self.num_nodes_list.append(num_nodes)
      if (len(self.num_nodes_list)==1) :
        self.weights.append(tf.Variable(np.random.randn(self.input_dimension, num_nodes)))
        self.biases.append(tf.Variable(np.random.randn(1,num_nodes)))
      else:
        self.weights.append(tf.Variable(np.random.randn(self.num_nodes_list[len(self.num_nodes_list)-2], num_nodes)))
        self.biases.append(tf.Variable(np.random.randn(1,num_nodes)))
      self.transfer_function.append(transfer_function)

      """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
      """

    def get_weights_without_biases(self, layer_number):
        return self.weights[layer_number]
        
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """

    def get_biases(self, layer_number):
        return self.biases[layer_number]
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """

    def set_weights_without_biases(self, weights, layer_number):
        self.weights[layer_number] = weights
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """

    def set_biases(self, biases, layer_number):
        self.biases[layer_number] = biases
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """

    def calculate_loss(self, y, y_hat):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_hat, name=None)
        mean_loss = tf.reduce_mean(loss)
        return mean_loss
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """

    def predict(self, X):
        for j in range(0,len(self.weights)):
          if j == 0 :
            a = tf.add(tf.matmul(X, self.weights[j]) , self.biases[j])
            if self.transfer_function[0] == "Relu":
              a= tf.nn.relu(a)
            elif self.transfer_function[0] == "Sigmoid":
              a= tf.nn.sigmoid(a)
          else:
            a = tf.add(tf.matmul(a, self.weights[j]) , self.biases[j])
            if self.transfer_function[j] == "Relu":
              a= tf.nn.relu(a)
            if self.transfer_function[j] == "Sigmoid":
              a= tf.nn.sigmoid(a)
        return a

        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        for epochs in range(num_epochs):
          for j in range(0 , X_train.shape[0], batch_size):
            with tf.GradientTape(persistent= True) as tape:
              predictions = self.predict(X_train[j:j+batch_size])
              loss = self.calculate_loss(y_train[j:j+batch_size], predictions)
            for i  in range(0, len(self.weights)):
                dloss_dw, dloss_db = tape.gradient(loss, [self.weights[i],self.biases[i]])
                self.weights[i].assign_sub(alpha * dloss_dw)
                self.biases[i].assign_sub(alpha * dloss_db)
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """

    def calculate_percent_error(self, X, y):
        error = 0
        current_net = self.predict(X)
        indexes = np.argmax(current_net, axis= -1)
        for i in range(0,len(y)):
          if indexes[i] != y[i]:
            error += 1
        percent_error = (error / X.shape[0])
        print(percent_error)
        return percent_error

        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """

    def calculate_confusion_matrix(self, X, y):
        predictions= self.predict(X)
        indexes = np.argmax(predictions, axis= -1)
        return tf.math.confusion_matrix(y,indexes)
        
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
