# Matta, Yashasvi
# 1002_091_131
# 2022_11_13
# Assignment_04_01

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import InputLayer
from cnn import CNN
from tensorflow.keras.datasets import cifar10
import numpy as np
def test_train():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    (X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
    number_of_train_samples_to_use = 100
    X_train = X_train[0:number_of_train_samples_to_use, :]
    Y_train = Y_train[0:number_of_train_samples_to_use]
    X_test = X_test[0:number_of_train_samples_to_use, :]
    Y_test = Y_test[0:number_of_train_samples_to_use]
    test_model= keras.models.Sequential()
    test_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    test_model.add(layers.MaxPooling2D((2, 2)))
    test_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    test_model.add(layers.Flatten())
    test_model.add(layers.Dense(64, activation='relu'))
    test_model.add(layers.Dense(10, activation='softmax'))
    test_model.compile(optimizer='SGD',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    history = test_model.fit(X_train, Y_train,batch_size=30, epochs=10)
    a_seq,b_seq = test_model.evaluate(X_test,Y_test)
    test_model = CNN()
    (X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
    number_of_train_samples_to_use = 100
    X_train = X_train[0:number_of_train_samples_to_use, :]
    Y_train = Y_train[0:number_of_train_samples_to_use]
    X_test = X_test[0:number_of_train_samples_to_use, :]
    Y_test = Y_test[0:number_of_train_samples_to_use]
    test_model.add_input_layer(shape=(32,32,3),name="input")
    test_model.append_conv2d_layer(32, 3, activation='relu')
    test_model.append_maxpooling2d_layer(pool_size=2,name="pool1")
    test_model.append_conv2d_layer(64, 3, activation='relu',name="conv2")
    test_model.append_flatten_layer(name="flat1")
    test_model.append_dense_layer(64, activation='relu',name="dense1")
    test_model.append_dense_layer(10, activation='softmax',name="dense2")
    test_model.set_loss_function("SparseCategoricalCrossentropy")
    test_model.set_metric('accuracy')
    test_model.set_optimizer(optimizer='SGD')
    out= test_model.train(X_train, Y_train, batch_size=30,num_epochs=10)
    a_cnn,b_cnn = test_model.evaluate(X_test,Y_test)
    print(a_seq,a_cnn,"\n",b_seq,b_cnn)
    # assert a_seq == a_cnn
    assert np.allclose(b_seq,b_cnn,atol=1e-1,rtol=1e-1)