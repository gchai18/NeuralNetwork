# NeuralNetwork

This project implements a multi-layer perceptron that trains on data sets using backpropagation. This neural network can be run using `Main.java`, which prompts the user for the name of the input file with network data and output file to print training data to and trains the network.

# Input File Data
The input file contains the information needed to create and train the network. `input.txt` is a sample of what the input data for NeuralNetwork.java looks like.
Each value in a line is separated by a space.

The first line contains an integer `M`, which describes the number of hidden layers in the neural network.
The second line contains an integer `N`, followed by `M` integers, followed by an integer `K`,
which are number of nodes in the input, hidden, and output layers of the neural network, respectively.
The third line contains 2 doubles, which are the minimum and maximum values for the randomized values of the weights.
The next line contains the data used to train the neural network. The first two doubles represent the learning factor lambda and
the amount the learning factor changes, the next integer contains the maximum number of times the network is run, and the last double is the minimum error threshold.
The next line contains an integer, `S`, that represents the number of sets of input values for the neural network.
The next `S` lines each contain `N` doubles; each line makes up a set of input values.
The next `K` lines contain `S`doubles, which are the target values for the output nodes. In this case, they are the values for OR, AND, and XOR.

# Output File
The output file contains the results of network training. `output.txt` is a sample of what the training data might look like. The output file first lists the number of times backpropagation was used and the final error after training. Then, for each input set, the output file lists the inputs, error, target values, and final outputs after training. It then gives the reason for why training was ended.
