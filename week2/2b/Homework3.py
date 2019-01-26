## Homework 3
##
## simple MNIST classifier network
##
## NSC3270/5270 Spring 2019

import numpy as np
import matplotlib.pyplot as plt

# supress unnecessary warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#########################################################################################
##
## load and format mnist images and labels
##

# load mnist
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# display some digits
fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# check out dimensions and types of mnist data
print('Training images shape: ', train_images.shape)
print('Training images type:  ', type(train_images[0][0][0]))
print('Testing images shape:  ', test_images.shape)
print('Testing images type:   ', type(test_images[0][0][0]))

# image shape
sz = train_images.shape[1]

# need to reshape and preprocess the training/testing images
train_images_vec = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
train_images_vec = train_images_vec.astype('float32') / 255
test_images_vec = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
test_images_vec = test_images_vec.astype('float32') / 255

# display new input dimensions/type
print('Training images shape: ', train_images_vec.shape)
print('Training images type:  ', type(train_images_vec[0][0]))
print('Testing images shape:  ', test_images_vec.shape)
print('Testing images type:   ', type(test_images_vec[0][0]))

# check out dimensions and types of mnist data
print('Training labels shape: ', train_labels.shape)
print('Training labels type:  ', type(train_labels[0]))

# also need to categorically encode the labels
print("First 5 training labels as labels:\n", train_labels[:5])
from keras.utils import to_categorical
train_labels_onehot = to_categorical(train_labels)
test_labels_onehot = to_categorical(test_labels)
print("First 5 training labels as one-hot encoded vectors:\n", train_labels_onehot[:5])

# display new output dimensions/type
print('Training labels shape (one hot): ', train_labels_onehot.shape)
print('Training labels type (one hot):  ', type(train_labels_onehot[0][0]))

#########################################################################################
##
## define and train neural network - we will discuss details of these keras pieces later
##

# import tools for basic keras networks 
from keras import models
from keras import layers

nout = 10
# create architecture of simple neural network model
# input layer  : 28*28 = 784 input nodes
# output layer : 10 (nout) output nodes
network = models.Sequential()
network.add(layers.Dense(nout, activation='sigmoid', input_shape=(sz * sz,)))

# print a model summary
print(network.summary())
print()
for layer in network.layers:
    print('layer name : {} | input shape : {} | output shape : {}'.format(layer.name, layer.input.shape, layer.output.shape))
print()
for layer in network.layers:
    print(layer.get_config())
print()

# compile network
network.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

# now train the network
history = network.fit(train_images_vec, train_labels_onehot, verbose=False, validation_split=.1, epochs=20, batch_size=128)
print('Done training!')

# test network
test_loss, test_acc = network.evaluate(test_images_vec, test_labels_onehot, verbose=False)
print('test_acc:', test_acc)

#########################################################################################
##
## some pieces needed to complete Homework 3
##

# get learned network weights and biases
W = network.layers[0].get_weights()[0]     # weights input to hidden
B = network.layers[0].get_weights()[1]     # bias to hidden
print('W {} | B {}'.format(W.shape, B.shape))

# model predictions (all 10000 test images)
out = network.predict(test_images_vec)

# model predictions (a single test image)
example = test_images_vec[123]
print(example.shape)
# vector passed to network.predict must be (?, 784)
example = example.reshape((1,example.shape[0]))
print(example.shape)

#########################################################################################
##
## Homework 3 Solution
##

##
## Q1. The original MNIST test_labels numpy array contains the digit value associated
## with the corresponding digit image (test_images). The output from the network (from
## out = network.predict(test_images_vec) above) contains the activations of the 10
## output nodes for every test image presented to the network. Write a function that
## takes the (10000,10) numpy array of output activations (of type float32) and returns 
## a (10000,) numpy array of discrete digit classification by the network (of type uint8).
## In other words, create a test_decisions numpy array of the same size and type as the
## MNIST test_labels array you started with. Below you will use both arrays to pull out
## test images that the network classifies correctly or incorrectly.
##
## To turn a numpy array of continuous output activations into a discrete digit classification,
## just take the maximum output as the "winner" that take all, determining the classification.
##
## In your function, feel free to use for loops. We are looking to see that you understand
## how to use the outputs generated by the network, not whether you can program using the
## most efficient python style.
##


##
## Q2. Comparing the correct answers (test_labels) and network classifications (test_decisions),
## for each digit 0..9, find one test image (test_image) that is classified by the network
## correctly and one test image that is classified by the network incorrectly. 
##
## Create a 2x10 plot of digit images (feel free to adapt the code above that uses subplot), with a 
## column for each digit 0..9 with the first row showing examples correctly classified (one example 
## for each digit) and the second row showing the examples incorrectly classified (one example 
## for each digit). Each subplot title should show the answer and the classification response 
## (e.g., displaying 4/2 as the title, if the correct answer is 4 and the classification was 2).
##


##
## Q3. Create "images" of the connection weight adapting the code used to display
## the actual digit images. There should be 10 weight images, an image for each
## set of weight connecting the input layer (784 inputs) to each output node.
## You will want to reshape the (784,1) vector of weights to a (28,28) image and
## display the result using imshow()


##
## Q4. Use the weight matrix (W), bias vector (B), and activation function (simple sigmoid)
## to reproduce in your own code the outputs (out) generated by the network (from
## this out = network.predict(test_images_vec))
##
## The simple sigmoid activation function is defined as follows:
## f(x) = 1 / (1+exp(-x))
##
## Feel free to use for loops or vector/matrix operations (we will go over the latter in
## in the coming weeks)
##
## Confirm that your output vectors and the keras-produced output vectors are the same
## (within some small epsilon since floating point calculations will often not come out
## exactly the same on computers).
##


