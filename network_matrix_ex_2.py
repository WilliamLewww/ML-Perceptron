#Based Off Original Code from:
#                 http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
import math

def problem(value):
    return 1 + math.sin((math.pi / 4) * value)

def logsig(value, derivative = 0):
    if derivative == 1:
        return value * (1 - value)
    
    return 1 / (1 + np.exp(-value))
    
input_case = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
output_case = np.array([[0,1,1,0]]).T

#4 Layer Network - [1 Input     2 Hidden      1 Output]
#                  -3 Nodes    -4 Neurons    -1 Neuron
#                              -4 Neurons

synapse_0 = 2*np.random.random((3,4)) - 1
synapse_1 = 2*np.random.random((4,4)) - 1
synapse_2 = 2*np.random.random((4,1)) - 1
for j in xrange(6000):
    #feedforward
    layer_0 = logsig(np.dot(input_case, synapse_0))
    layer_1 = logsig(np.dot(layer_0, synapse_1))
    layer_2 = logsig(np.dot(layer_1, synapse_2))
    
    #backprop
    layer_2_delta = (output_case - layer_2) * logsig(layer_2, 1)
    layer_1_delta = layer_2_delta.dot(synapse_2.T) * logsig(layer_1, 1)
    layer_0_delta = layer_1_delta.dot(synapse_1.T) * logsig(layer_0, 1)
    synapse_2 += layer_1.T.dot(layer_2_delta)
    synapse_1 += layer_0.T.dot(layer_1_delta)
    synapse_0 += input_case.T.dot(layer_0_delta)
    
print layer_2