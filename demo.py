from network_matrix import *

#4 Layer Network - [1 Input     2 Hidden      1 Output]
#                  -2 Nodes    -4 Neurons    -1 Neuron
#                              -4 Neurons

input_case = generate_array([[0,0],[0,1],[1,0],[1,1]])
output_case = generate_array([[1,0,0,-1]], 1)

synapse_list = generate_synapse([2, 4, 4, 1])
activation_list = generate_activation([1, 1, 2])
print(feedforward_and_backpropagate(6000, synapse_list, input_case, output_case, activation_list))