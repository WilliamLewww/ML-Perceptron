import math
import numpy as np

#for -2 <= p <= 2
def problem(value):
    return 1 + math.sin((math.pi / 4) * value)

def purelin(value, type = 0):
    if type == 0:
        return value

    return 1

def logsig(value, type = 0):
    if type == 0:
        temp_matrix = np.matrix(value)
        for x in np.nditer(temp_matrix, op_flags=['readwrite']):
            x[...] = (1 / (1 + (math.e ** -x)))

        return temp_matrix

    return value * (1 - value)

def jacobian(values):
    temp_matrix = np.zeros(shape=(len(values), len(values)))
    np.fill_diagonal(temp_matrix, values)

    return temp_matrix

#Neural Network Design(11-14) by [Martin T. Hagan, Howard B. Demuth, Mark Hudson Beale, Orlando De Jesus]
class Network:
    def __init__(self):
        self.a_0 = np.matrix([[1]])
        self.learning_rate = 0.1

        self.w_1 = np.matrix([[-0.27], [-0.41]])
        self.b_1 = np.matrix([[-0.48], [-0.13]])
        self.w_2 = np.matrix([[0.09, -0.17]])
        self.b_2 = np.matrix([[0.48]])

    def feedforward_and_backpropagate(self):
        #a = f(Wa + b)
        a_1 = logsig((self.w_1 * self.a_0) + self.b_1)
        a_2 = purelin((self.w_2 * a_1) + self.b_2)

        #e = t - a
        e = problem(1) - a_2

        s_2 = -2 * purelin(1, 1) * e
        s_1 = jacobian([logsig(a_1[0], 1), logsig(a_1[1], 1)]) * self.w_2.reshape(2,1) * s_2

        self.w_2 = self.w_2 - (self.learning_rate * s_2 * a_1.reshape(1,2))
        self.b_2 = self.b_2 - (self.learning_rate * s_2)

        self.w_1 = self.w_1 - (self.learning_rate * s_1 * self.a_0)
        self.b_1 = self.b_1 - (self.learning_rate * s_1)

        return e

network = Network()
print network.feedforward_and_backpropagate()
print network.feedforward_and_backpropagate()
print network.feedforward_and_backpropagate()
