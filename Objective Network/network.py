import math
import random

class Neuron:
    #value, synapseList, function
    def __init__(self, value, bias, function):
        self.value = value
        self.function = function

        self.bias = bias

        self.synapseList = []

    def connect(self, other, weight):
        self.synapseList.append(Synapse(self, other, weight))

    def activation(self):
        if self.function == 1:
            self.value = (1 / (1 + (math.e ** -self.value)))

        if self.function == 2:
            self.value = ((math.e ** self.value) - (math.e ** -self.value)) / ((math.e ** self.value) + (math.e ** -self.value))

        if self.function == 3:
            if self.value < 0:
                self.value = 0
        """else:
                self.value = self.value"""

    def de_activation(self):
        if self.function == 1:
            return self.value * (1 - self.value)

        if self.function == 2:
            return (1 - (((math.e ** self.value) - (math.e ** -self.value)) / ((math.e ** self.value) + (math.e ** -self.value)) ** 2))

        if self.function == 3:
            if self.value > 0:
                return 1
            else:
                return 0

        return 1

class Synapse:
    #neuronA, neuronB, weight
    def __init__(self, a, b, weight):
        self.neuronA = a
        self.neuronB = b

        self.weight = weight

class Layer:
    #neuronList
    def __init__(self, count, function):
        self.neuronList = []
        for x in range(count):
            self.neuronList.append(Neuron(0, 0, function))

    def connect_layer(self, other):
        for x in range(len(self.neuronList)):
            for y in range(len(other.neuronList)):
                self.neuronList[x].connect(other.neuronList[y], ((random.random() * 20) - 10) / 10)

    def activate_layer(self):
        for neuron in self.neuronList:
            neuron.activation()

    def return_neurons(self):
        valueList = []
        for neuron in self.neuronList:
            valueList.append(neuron.value)

        return valueList

    def return_bias(self):
        biasList = []
        for neuron in self.neuronList:
            biasList.append(neuron.bias)

        return biasList

class Network:
    #layerList, expectedOutput, errorList
    def __init__(self):
        self.layerList = []
        self.errorList = []
        self.learningRate = 0.01

        self.isConnected = 0

    def clear(self):
        self.__init__()

    def generate_example(self):
        """ 4 Layer Network
            2 N's in Input Layer
            3 N's in Hidden Layer {1}
            5 N's in Hidden Layer {2}
            2 N's in Output Layer """

        #add layers to network
        self.append_layer(2,1)
        self.append_layer(3,1)
        self.append_layer(5,1)
        self.append_layer(1,1)

        #connect all synapses for every neuron in each layer
        self.fully_connect()

        #set the value of the input neurons (first layer)
        self.feed_input([1, 0], [1])

        #forward propagate the network
        #self.feedforward()

        #write the values of each neuron in the console
        #self.print_layers()

    def append_layer(self, count, function):
        self.layerList.append(Layer(count, function))

    def fully_connect(self):
        for x in range(len(self.layerList) - 1):
            self.layerList[x].connect_layer(self.layerList[x + 1])

        for x in range(len(self.layerList[len(self.layerList) - 1].neuronList)):
            self.errorList.append(1.0)

        self.isConnected = 1

    def print_layers(self, output = 0):
        if output == 0 or output == 1:
            print("Network:")
            for layer in self.layerList:
                print(layer.return_neurons())

        if output == 2:
            print("Bias:")
            for layer in self.layerList:
                print(layer.return_bias())

        if output == 0 or output == 3:
            print("Error:")
            print(self.errorList)

    def feed_input(self, input, expectedOutput = [0]):
        self.expectedOutput = expectedOutput
        for x in range(len(input)):
            self.layerList[0].neuronList[x].value = input[x]

        for x in range(len(self.errorList)):
            self.errorList[x] = 1.00

    def run_till_error(self, error, output = 0):
        hasError = 1
        iterations = 0
        while hasError == 1:
            hasError = 0

            for e in self.errorList:
                if abs(e) > error / 100:
                    hasError = 1

            self.feedforward()
            self.backpropagate()
            if output == 1:
                self.print_layers()

            iterations += 1

        return iterations

    def feedforward(self):
        if self.isConnected == 0:
            self.fully_connect()
            self.isConnected = 1

        for x in range(len(self.layerList) - 1):
            for neuron in self.layerList[x].neuronList:
                for synapse in neuron.synapseList:
                    synapse.neuronB.value += (synapse.neuronA.value * synapse.weight) + synapse.neuronB.bias
            self.layerList[x + 1].activate_layer()

        for x in range(len(self.layerList[len(self.layerList) - 1].neuronList)):
            self.errorList[x] = (self.expectedOutput[x] - self.layerList[len(self.layerList) - 1].neuronList[x].value)

        return self.errorList

    def backpropagate(self):
        total = 0.0
        layerTotal = 0.0

        for x in range(len(self.layerList) - 1, 0, -1):
            for y in range(len(self.layerList[x - 1].neuronList)):
                for z in range(len(self.layerList[x].neuronList)):
                    if x == len(self.layerList) - 1:
                        tempVal = -self.errorList[z] * self.layerList[x].neuronList[z].de_activation()
                        total += tempVal * self.layerList[x - 1].neuronList[y].synapseList[z].weight
                        self.layerList[x - 1].neuronList[y].synapseList[z].weight -= self.learningRate * (tempVal * self.layerList[x - 1].neuronList[y].value)
                    else:
                        tempVal = layerTotal * self.layerList[x].neuronList[z].de_activation()
                        total += tempVal * self.layerList[x - 1].neuronList[y].synapseList[z].weight
                        self.layerList[x - 1].neuronList[y].synapseList[z].weight -= self.learningRate * (tempVal * self.layerList[x - 1].neuronList[y].value)
            layerTotal += total
            total = 0
