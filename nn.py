import csv
import math
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py
from random import random
from random import seed

LEARNING_RATE   = 0.5
EPOCHS          = 20

# initialise and return the neural network
# with random numbers between (0, 1)
def init_network(num_inputs, num_hidden, num_outputs):
    network = list()

    # hidden has input + 1, one for input from ds, one additional for bias
    hidden_layer = [{'weights': [random() for i in range(num_inputs + 1)]} for i in range(num_hidden)]
    network.append(hidden_layer)

    # output has 1 weight plus bias in linear instance
    output_layer = [{'weights': [random() for i in range(num_hidden + 1)]} for i in range(num_outputs)]
    network.append(output_layer)

    return network

# calculate the neuron activation for an input
# nb. assumes bias is LAST weight in the list of weights (potential for adjusting)?
def activate(weights, inputs):
    # print("weights: ")
    # print(weights)
    # print("inputs: ")
    # print(inputs)
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# specific transfer neuron activation fn
def transfer(activation):
    return activation / (1 + math.fabs(activation))

# specific derivative
def transfer_derivative(output):
    return (1.0 / (1 + math.fabs(output))**2)

# # his transfer neuron activation fn
# def transfer(activation):
#     return 1.0 / (1.0 + math.exp(-activation))
#
# # Calculate the derivative of an neuron output
# def transfer_derivative(output):
# 	return output * (1.0 - output)

# modified such that output layer is linear
def forward_propagate(network, row):
    inputs = row
    for i, layer in enumerate(network):
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            if i != len(network):
                neuron['output'] = transfer(activation)
            else:
                neuron['output'] = activation
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        # check if hidden layer
        if i != len(network)-1:
            # for each neuron in the hidden layer(s)
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    # delta_i = sum (weights_ki * delta_k)
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)

        # else it is output layer
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                # print(expected[j])
                error = 0.5*(math.fabs(expected[j][0] - neuron['output']))**2
                errors.append(error)

        # multiply each layer by the tranfer derivative fn
        for j in range(len(layer)):
            neuron = layer[j]
            # print("this hip hop happening")
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# def update_weights(network, init_input):
#     for i in range(len(network)):
#         inputs = init_input
#         if i != 0:
#             inputs = [neuron['output'] for neuron in network[i-1]]
#         for neuron in network[i]:
#             for j in range(len(inputs)-1):
#                 neuron['weights'][j] += LEARNING_RATE * neuron['delta'] * inputs[j]
#             neuron['weights'][-1] += LEARNING_RATE * neuron['delta']

def update_weights(network):
    for i in range(len(network)):
        # inputs = init_input
        # if i != 0:
        #     inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(neuron['weights'])-1):
                neuron['weights'][j] += LEARNING_RATE * neuron['delta']
            neuron['weights'][-1] += LEARNING_RATE * neuron['delta']

# online learning not batch learning as errors not accumulated across an epoch before updating - modify?
# probably going to have to write this fn from scratch
def train_network(network, input_vals, expected):
    for epoch in range(EPOCHS):
        sum_error = 0
        # print(len(expected))
        output_vals = []
        for row in input_vals:
            outputs = forward_propagate(network, row)
            output_vals.append(outputs)
            # expected = [0 for i in range(num_outputs)]
            # expected[row[-1]] = 1
            backward_propagate_error(network, expected)
        # print(len(output_vals))
        sum_error += sum([(expected[i][0] - output_vals[i][0])**2 for i in range(len(expected))])
        update_weights(network)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, LEARNING_RATE, sum_error))

def numerically_estimated_pd(network, epsilon):
    w = []
    outputs = []
    for neuron in network:
        neuron = neuron[0]
        w.append(neuron['weights'])
        outputs.append(neuron['output'])
    # print(w)

    e = [0 for i in range(len(network))]

def nn(input_vals, expected):
    seed(1)

    network = init_network(1, 1, 1)
    # for i, row in enumerate(input_vals):
    #     forward_propagate(network, row)
    #     backward_propagate_error(network, expected[i])

    train_network(network, input_vals, expected)

    for layer in network:
        print(layer)

    # numerically_estimated_pd(network, 0.1)
