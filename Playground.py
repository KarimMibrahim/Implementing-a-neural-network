#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:18:53 2017

@author: KarimM
"""

import numpy as np 
from random import random

# Initializing the network
def initialize_network(*args):
    network = list()
    for layer_number in range(len(args)-1):
        layer = [{'weights':[random() for i in range(args[layer_number] + 1)]} for i in range(args[layer_number+1])]
        network.append(layer)
    return network

# ReLu Activation Function
def relu(input):
    return np.maximum(input, 0)

# ReLu derivative [[[Make it applicable on vectors]]]
def relu_diff(input):
    if input < 0:
        return 0 
    return 1
        

#Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = relu(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs



# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])