import numpy
# Imports sigmoid function, expit()
import scipy.special

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.weight_input_layer_to_hidden_layer = numpy.random.normal(0.0, pow(input_nodes, -0.5), (hidden_nodes, input_nodes))
        self.weight_hidden_layer_to_output_layer = numpy.random.normal(0.0, pow(hidden_nodes, -0.5), (output_nodes, hidden_nodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weight_input_layer_to_hidden_layer, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hidden_layer_to_output_layer, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weight_hidden_layer_to_output_layer.T, output_errors)

        self.weight_hidden_layer_to_output_layer += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.weight_input_layer_to_hidden_layer += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2) .T
        hidden_inputs = numpy.dot(self.weight_input_layer_to_hidden_layer, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hidden_layer_to_output_layer, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

myNeuralNetwork = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# myNeuralNetwork.query([1.0, 0.5, -1.5])

training_data_file = open("./data/train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Number of times to run training data
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        myNeuralNetwork.train(inputs, targets)
        pass
    pass

test_data_file = open("./data/test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# How well neural network performs
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = myNeuralNetwork.query(inputs)

    label = numpy.argmax(outputs)

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = numpy.asarray(scorecard)
print("Perfmance  = ", scorecard_array.sum() / scorecard_array.size)
