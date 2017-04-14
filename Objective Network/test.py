from network import *

network = Network()
network.append_layer(2,1)
network.append_layer(10,1)
network.append_layer(1,1)
network.fully_connect()

running = 1
while running:
    with open("cases.txt") as file:
        for line in file:
            fileInput = line.split()
            network.feed_input([int(fileInput[0]), int(fileInput[1])], [int(fileInput[2])])
            print(network.run_till_error(15))
