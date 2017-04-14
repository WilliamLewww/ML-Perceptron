from network import *

network = Network()
running = 1

def edit_neuron(layer, neuron, type):
    if type == 1:
        print("Bias:")
        a = float(input())
        network.layerList[layer].neuronList[neuron].bias = a

    if type == 2:
        print("Activation Function:")
        a = int(input())
        network.layerList[layer].neuronList[neuron].function = a

def process_input(type):
    if type == 0:
        network.clear()

    if type == 1:
        network.print_layers()
        print()
        
        print("Neuron Count, Activation Function:")
        a = [int(x) for x in input().split()]
        network.append_layer(a[0], a[1])

    if type == 2:
        network.fully_connect()

    if type == 3:
        network.print_layers()
        print()

        print("Input:")
        a = [float(x) for x in input().split()]
        print("Expected Output:")
        b = [float(x) for x in input().split()]

        network.feed_input(a, b)

    if type == 4:
        network.feedforward()
        network.backpropagate()
        network.print_layers()

    if type == 5:
        network.run_till_error(10, 1)

    if type == 6:
        network.generate_example()

    if type == 7:
        network.print_layers()

    if type == 8:
        network.print_layers()
        print()

        print("Select Layer and Neuron")
        a = [int(x) for x in input().split()]
        print("Select Function")
        b = int(input())

        edit_neuron(a[0], a[1], b)

def clear_screen():
    for x in range(25):
        print()

clear_screen()

while running == 1:
    print("[      0 - Clear Network                  ]")
    print("[      1 - Append Layer                   ]")
    print("[      2 - Fully Connect                  ]")
    print("[      3 - Set Input and Expected Output  ]")
    print("[      4 - Run Once                       ]")
    print("[      5 - Run Until Error                ]")
    print("[      6 - Generate Example               ]")
    print("[      7 - Print Network                  ]")
    print("[      8 - Edit Network                   ]")
    print()
    print("Enter a command:")
    ip = int(input())

    clear_screen()
    process_input(ip)

    print()
