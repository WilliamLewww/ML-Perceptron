#Demonstration of network_matrix module
#by William Lew

from network_matrix import *

#suppress scientific notation for easier readability
np.set_printoptions(suppress = True)

#4 Layer Network - [1 Input     2 Hidden      1 Output]
#                  -2 Nodes    -4 Neurons    -1 Neuron
#                              -4 Neurons

input_case = generate_array([[0,0],[0,1],[1,0],[1,1]])
output_case = generate_array([[0,5,5,10]], 1)

synapse_list = generate_synapse([2, 4, 1])
activation_list = generate_activation([1, 0])
print(feedforward_and_backpropagate(6000, synapse_list, input_case, output_case, 0.15, activation_list))
print(feedforward(input_case, synapse_list, activation_list))
print(input_case)
