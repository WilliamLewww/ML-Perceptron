#Fully Connected Recurrent Neural Network
#by William Lew

#Based Off Original Code from:
#                 http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
import math

def __logsig(value, derivative = 0):
    if derivative == 1:
        return value * (1 - value)
    
    return 1 / (1 + np.exp(-value))
 
def __purelin(value, derivative = 0):
    if derivative == 1:
        return 1
    
    return value
    
def __tanh(value, derivative = 0):
    if derivative == 1:
        return 1 - (np.tanh(value) ** 2)
    
    return np.tanh(value)
    
def generate_array(array, output = 0):
    if output == 1:
        return np.array(array).T
    return np.array(array)
    
def generate_synapse(array):
    temp_list = []
    for x in range(1, len(array)):
        temp_list.append(2*np.random.random((array[x - 1], array[x])) - 1)
    
    return temp_list

def generate_activation(array):
    temp_list = []
    for x in array:
        if x == 0:
            temp_list.append(__purelin)
        if x == 1:
            temp_list.append(__logsig)
        if x == 2:
            temp_list.append(__tanh)
    
    return temp_list
    
def feedforward_and_backpropagate(loop_count, synapse_list, input_case, output_case, activation_list = None):
    if activation_list == None:
        activation_list = [__logsig for x in range(len(synapse_list))]
        
    for j in xrange(loop_count):
        #feedforward
        layer_list = [input_case]
        for x in range(len(synapse_list)):
            layer_list.append(activation_list[x](np.dot(layer_list[x], synapse_list[x])))
        
        #backprop
        layer_delta_list = [(output_case - layer_list[len(layer_list) - 1]) * activation_list[len(activation_list) - 1](layer_list[len(layer_list) - 1], 1)]
        for x in range(len(synapse_list) - 2, -1, -1):
            layer_delta_list.append(layer_delta_list[len(layer_delta_list) - 1].dot(synapse_list[x + 1].T) * activation_list[x](layer_list[x + 1], 1))
        
        for x in range(len(synapse_list)):
            synapse_list[x] += layer_list[x].T.dot(layer_delta_list[len(synapse_list) - x - 1])
            
    return layer_list[len(layer_list) - 1]