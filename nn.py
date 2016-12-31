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

LEARNING_RATE   = -0.1
EPOCHS          = 60
NUM_H_NEURONS   = 20

EPSILON         = 10**(-6)

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

def test_network(network, inputs, expected):
    nn_outputs = []
    for row in inputs:
        row_output = forward_propagate(network, row)
        nn_outputs.append(row_output)
    mse_val = mse(nn_outputs, expected)
    return mse_val

# calculate the mse given empirical / output vals and the expected vals
def mse(output_vals, expected):
    sum_error = 0
    sum_error += sum([0.5*(math.fabs(expected[i][0] - output_vals[i][0]))**2 for i in range(len(expected))])
    return sum_error

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
def train_network(network, train_vals, train_expected, test_vals, test_expected):
    train_mse_vals = []
    test_mse_vals = []
    # for each learning epoch
    for epoch in range(EPOCHS):
        # sum_error = 0
        output_vals = []

        # forward propagate and then back prop with the train values
        for row in train_vals:
            outputs = forward_propagate(network, row)
            output_vals.append(outputs)

            backward_propagate_error(network, train_expected)

        # calculate the mse on the training values
        train_mse = mse(output_vals, train_expected)

        # caluculate the mse on the test values
        test_mse = test_network(network, test_vals, test_expected)

        # update the weights in the network
        update_weights(network)

        train_mse_vals.append(train_mse)
        test_mse_vals.append(test_mse)

        print('>epoch=%d, lrate=%.3f, error training=%.3f, error validation=%.3f' % (epoch, LEARNING_RATE, train_mse, test_mse))

    return (train_mse_vals, test_mse_vals)

def numerically_estimated_pd(network):
    w = []
    outputs = []
    for neuron in network:
        neuron = neuron[0]
        w.append(neuron['weights'])
        outputs.append(neuron['output'])
    # print(w)

    e = [0 for i in range(len(network))]

def error_plot(trainmse, testmse):
    x = [i for i in range(EPOCHS)]

    # fig = plt.figure(121)


    tr_y = [math.log(y) for y in trainmse]
    te_y = [math.log(y) for y in testmse]

    plt.plot(x, tr_y, color='r', label='training mse')

    plt.plot(x, te_y, color='g', label='testing mse')
    plt.title('Plot of Learning epoch again the MSE for training and testing datasets')

    plt.xlabel('learning epoch')
    plt.ylabel('log MSE')
    # plt.yscale('log')

    plt.legend()

    # plt.xlim((0, max(x_val)))
    # plt.ylim((min(y_val)-0.25, 5.25))

    plt.grid(True)
    plt.show()

def notzero(x):
    if x != 0:
        return x

def sinc(x):
    return (math.sin(x) / x)

def compare_fn(network):
    xs = [i/100 for i in range(-1000, 1005, 5)]
    xs.remove(0.0)
    sinc_vals = [sinc(x) for x in xs]

    list_xs = [[x] for x in xs]
    nn_vals = []

    for row in list_xs:
        row_output = forward_propagate(network, row)
        nn_vals.append(row_output)

    nn_vals = [nnv[0] for nnv in nn_vals]

    plt.plot(xs, sinc_vals, color='b', label='sinc(x)')

    plt.plot(xs, nn_vals, color='r', label='output from neural network')

    plt.title('Plot of sinc(x) and trained neural network trying to model sinc(x)')

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.yscale('log')

    plt.legend()
    plt.grid(True)
    plt.show()

def nn(train_vals, train_expected, test_vals, test_expected):
    seed(1)

    # LEARNING_RATE = lrate

    network = init_network(1, NUM_H_NEURONS, 1)

    train_mse_vals, test_mse_vals = train_network(network, train_vals, train_expected, test_vals, test_expected)

    lrates = [-0.05, -0.1, 1, 5]

    error_plot(train_mse_vals, test_mse_vals)

    return (network, train_mse_vals, test_mse_vals)


    # for layer in network:
    #     print(layer)

    # numerically_estimated_pd(network, 0.1)
