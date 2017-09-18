#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:42:39 2017

@author: KarimM
"""

"""
This code includes different pieces of building a neural network: 
    - init_net: initializes a network with the given number of layers and nodes
    - forward_prop: passes an input through the network and returns the outputs of the network
    - back_prop: performs back propoagation on the passed network for a given input and expected output and updates the weights of the network accordingly
    - train: train a given network for on a given dataset through back propagation
    - saveNet & loadNet: saves and loads a given network to the hard disk
    - testingNetwork: evaluates the performance of the network on a given testset and return the accuracy
"""


import numpy as np
from numpy import genfromtxt
import pickle
#import matplotlib.pyplot as plt

""" 
Activation Funcations
"""

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_diff(z):
    return z * (1.0 - z)

def softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

# ReLu Activation Function
def relu(z):
    return np.maximum(z, 0.0)

# ReLu derivative [[[Make it applicable on vectors]]]
def relu_diff(z):
    z[z<=0] = 0
    z [z > 0] = 1
    return z
"""
Initializing the network
"""
def init_net(*args):
    network = list()
    for layer_number in range(1,len(args)):
        network.append(np.random.rand(args[layer_number],args[layer_number-1]+1)) # adding 1 for the bias term
    return network
    
"""
forward propagation
"""
def forward_prop(net,inputs):
    inputs = np.append(inputs,1)
    Nodes_state = []
    for counter in range(0,len(net)-1):
        if (counter == 0):
            H_hidden = relu(np.dot(net[counter],inputs))
        else: 
            H_hidden = relu(np.dot(net[counter],H_hidden))
        H_hidden = np.append(H_hidden,1)
        Nodes_state.append(H_hidden)
    Outputs = softmax(np.dot(net[-1],H_hidden))
    Nodes_state.append(Outputs)
    return Nodes_state

"""
back propagation
"""
def back_prop(network,expected,inputs,l_rate):
    Nodes_state = forward_prop(network,inputs)
    inputs = np.append(inputs,1)
    dE_dO = expected - Nodes_state[-1]
    #dE_dZo = dE_dO * sigmoid_diff(outputs) # For Sigmoid
    dE_dZ = dE_dO                          # For softmax
    for counter in range(len(network)-1,0,-1):
        hidden_output = Nodes_state[counter-1]
        dE_dW = np.dot(np.reshape(dE_dZ,[len(dE_dZ),1]),np.reshape(hidden_output,[1,len(hidden_output)])) # I want [dE_dZo] as a vertical vector dot produt by hidden outputs as horizontal -> this gives weight update for each weight
        dE_dZ = np.dot(dE_dZ,network[counter][:,:-1]) * relu_diff(hidden_output[:-1]) # hidden_output[:-1] to exclude the bias
        network[counter] += l_rate * dE_dW
    dE_dW = np.dot(np.reshape(dE_dZ,[len(dE_dZ),1]),np.reshape(inputs,[1,len(inputs)]))
    network[0] += l_rate * dE_dW

"""
Training the network
"""

def train(network,dataset,iterations,n_outputs,l_rate,testDataset, printAfter = 100):
    TrainErrors = list()
    TestErrors = list()
    TrainAccuracyList = list()
    TestAccuracyList = list()
    
    for i in range(iterations):
        Train_sum_error = 0.0
        Test_sum_error = 0.0
        for row in dataset:
            expected = [0 for k in range(n_outputs)]
            expected[row[-1]] = 1
            Train_sum_error += sum((expected - forward_prop(network,row[:-1])[-1])**2)
            back_prop(network,expected,row[:-1],l_rate)
        TrainErrors.append(Train_sum_error)
            
        # Cost for Test set
        for row in testDataset:
            expected = [0 for k in range(n_outputs)]
            expected[row[-1]] = 1
            Test_sum_error += sum((expected - forward_prop(network,row[:-1])[-1])**2)
        TestErrors.append(Test_sum_error)
        
        # printing errors every 100 (default) iterations
        if (i % printAfter == 0):
            print 'Train error is',Train_sum_error, 'Test error is', Test_sum_error
        
        # Testing classification accuracy on train and test datasets
        TrainAccuracy = testingNetwork(dataset,network)
        TestAccuracy = testingNetwork(testDataset,network)
        TrainAccuracyList.append(TrainAccuracy)
        TestAccuracyList.append(TestAccuracy)
        # Saving errors in files
        with open("TrainError.txt", 'a') as file_handler:
            file_handler.write("{0}\n".format(Train_sum_error))
        with open("TestError.txt", 'a') as file_handler:
            file_handler.write("{0}\n".format(Test_sum_error))
        with open("TrainAccuracy.txt", 'a') as file_handler:
            file_handler.write("{0}\n".format(TrainAccuracy))
        with open("TestAccuracy.txt", 'a') as file_handler:
            file_handler.write("{0}\n".format(TestAccuracy))

"""
Saving and Loading the net
"""
def saveNet(net,fileName):
    with open(fileName, "wb") as fp:   #Pickling
        pickle.dump(net, fp)

def loadNet(fileName):
    with open(fileName, "rb") as fp:   # Unpickling
        net = pickle.load(fp)
    return net

"""
Evaluation Section
"""

def predict(network, row):
    outputs = forward_prop(network, row)[-1]
    return np.argmax(outputs)

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def testingNetwork(dataSet,network):
    predictions = []
    for i in range(len(dataSet)):
        predictions.append(predict(network,dataSet[i][:-1]))
    actual = [row[-1] for row in dataSet]
    return accuracy_metric(actual,predictions)


"""
def plotResults(directory = '/Users/KarimM/Desktop/CloudOutputs/100_40/TrainAccuracy.txt'):
    TrainAccuracies = list()
    with open(directory) as f:
        for line in f:
            sTrainAccuracies.append(float(line))
    #plt.figure(figsize=(70, 70))
    plt.plot(TrainAccuracies,'b', label = "Training")
    plt.plot(TestAccuracies,'r', label = "Testing")
    plt.xlabel('Iteration')
    plt.ylabel('Classification Accuracy')
    plt.title('Classification Accuracy through iterations')
    plt.yticks(np.arange(40, 101.0, 5.0))
    plt.grid(True)
    plt.legend()
    #plt.show()
    plt.savefig('test.png',dpi = 1080)
"""

"""
The following is a script for training the network on the remote server
"""

"""
x_training = genfromtxt('../data/Question2_123/x_train.csv',delimiter = ",")
y_training = genfromtxt('../data/Question2_123/y_train.csv',delimiter = ",")
x_training = x_training.astype(int)
y_training = y_training.astype(int)
y_training = y_training.reshape([len(y_training),1])
TrainData = np.append(x_training, y_training, axis=1)  

x_test = genfromtxt("../data/Question2_123/x_test.csv",delimiter = ",")
y_test = genfromtxt("../data/Question2_123/y_test.csv",delimiter = ",")
x_test = x_test.astype(int)
y_test = y_test.astype(int)
y_test = y_test.reshape([len(y_test),1])
TestData = np.append(x_test, y_test, axis=1) 

network = init_net(14,28,28,28,28,28,28,4)
train(network,TrainData,1000,4,0.3,TestData,5)
saveNet(network,"network.txt")
"""

"""
x_training = genfromtxt("../../assignment1/Question2_123/x_train.csv",delimiter = ",")
y_training = genfromtxt("../../assignment1/Question2_123/y_train.csv",delimiter = ",")
x_training = x_training.astype(int)
y_training = y_training.astype(int)
y_training = y_training.reshape([len(y_training),1])
TrainData = np.append(x_training, y_training, axis=1)
"""


"""
x_test = genfromtxt("/Users/KarimM/Google Drive/PhD/Courses/Deep Learning/assignment1/Question2_123/x_test.csv",delimiter = ",")
y_test = genfromtxt("/Users/KarimM/Google Drive/PhD/Courses/Deep Learning/assignment1/Question2_123/y_test.csv",delimiter = ",")
x_test = x_test.astype(int)
y_test = y_test.astype(int)
y_test = y_test.reshape([len(y_test),1])
TestData = np.append(x_test, y_test, axis=1)     
"""

"""
layers = genfromtxt('/Users/KarimM/Google Drive/PhD/Courses/Deep Learning/assignment1/Question2_4/b/w-100-40-4nonames.csv',delimiter = ",")
biases = genfromtxt('/Users/KarimM/Google Drive/PhD/Courses/Deep Learning/assignment1/Question2_4/b/b-100-40-4nonames.csv',delimiter = ",")
GivenNetwork[0] = layers[0:14].T
GivenNetwork[1] = layers[14:114,0:40].T
GivenNetwork[2] = layers[114:154,0:4].T

GivenNetwork[0] = np.hstack((GivenNetwork[0],np.reshape(biases[0],[len(biases[0]),1])))
GivenNetwork[1] = np.hstack((GivenNetwork[1],np.reshape(biases[1][0:40],[len(biases[1][0:40]),1])))
GivenNetwork[2] = np.hstack((GivenNetwork[2],np.reshape(biases[2][0:4],[len(biases[2][0:4]),1])))

X=[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 3]
"""