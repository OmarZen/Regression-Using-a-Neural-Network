import numpy as np
class Weight:
    def __init__(self, value, neuron_from, neuron_to):
        self.value = value
        self.name = f"W{neuron_from.name},{neuron_to.name}"
class Neuron:
    def __init__(self, num_inputs, activation_function, layer_type, index):
        if layer_type == "input":
            self.name = f"in{index}"

        elif layer_type == "hidden":
            self.name = f"h{index}"

        elif layer_type == "output":
            self.name = f"out{index}"
        self.bias = np.random.rand()
        self.output = None
        self.delta = None
        self.activation_function = activation_function
        self.layer_type = layer_type
        self.weights = []  ########

    def connect(self, neuron):
        weight_value = np.random.rand()
        weight = Weight(weight_value, self, neuron)
        self.weights.append(weight)   #####################
        return self.weights

    # def calculate_output(self, inputs):
    #     # Calculate the weighted sum of inputs and apply the activation function
    #     weighted_sum = np.dot(inputs, self.weights) + self.bias
    #     self.output = self.activation_function(weighted_sum)
    #     return self.output

    def calculate_output(self, inputs):
        # Extract weights' values from Weight objects
        weights_values = np.array([weight.value for weight in self.weights])

        # Calculate the weighted sum of inputs and apply the activation function
        weighted_sum = np.dot(inputs, weights_values) + self.bias
        self.output = self.activation_function(weighted_sum)
        return self.output

    def calculate_derivative(self):
        # Calculate the derivative of the activation function
        return self.activation_function(self.output, derivative=True)



class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function, layer_type):
        self.neurons = [Neuron(num_inputs, activation_function, layer_type, i + 1) for i in range(num_neurons)]

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.learning_rate = None
        self.num_epochs = None

    def set_architecture(self, input_size, hidden_size, output_size, learning_rate, num_epochs):
        print("Step 3_A: Setting up the neural network...")
        # Set hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Input layer with sigmoid activation
        input_layer = Layer(input_size, num_inputs=input_size, activation_function=self.sigmoid, layer_type="input")
        self.layers.append(input_layer)

        # Hidden layer with sigmoid activation
        hidden_layer = Layer(hidden_size, num_inputs=input_size, activation_function=self.sigmoid, layer_type="hidden")
        self.layers.append(hidden_layer)

        # Output layer with linear activation for regression
        output_layer = Layer(output_size, num_inputs=hidden_size, activation_function=None, layer_type="output")
        self.layers.append(output_layer)

        # Connect neurons with weights
        for input_neuron in self.layers[0].neurons:
            for hidden_neuron in self.layers[1].neurons:
                input_neuron.connect(hidden_neuron)

        for hidden_neuron in self.layers[1].neurons:
            for output_neuron in self.layers[2].neurons:
                hidden_neuron.connect(output_neuron)
        print("Step 3_A: Setting up the neural network complete.")

    def check_architecture(self):
        print("Checking neural network architecture...")
        print(f"Number of layers: {len(self.layers)}")

        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i + 1}:")
            print(f"Number of neurons: {len(layer.neurons)}")

            for j, neuron in enumerate(layer.neurons):
                print(f"Neuron {neuron.name}:")
                print(f"Bias: {neuron.bias}")

                if neuron.weights:
                    print("Weights:")
                    for weight in neuron.weights:
                        print(f"Weight {weight.name}: Value = {weight.value}")
                else:
                    print("No weights connected to this neuron.")

        print("\nChecking complete.")

    def debug_set_architecture(self):
        print("Debugging set architecture...")

        # Print the architecture details
        print("Neural Network Architecture:")
        print(f"Number of layers: {len(self.layers)}")

        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i + 1}:")
            print(f"Number of neurons: {len(layer.neurons)}")

            for j, neuron in enumerate(layer.neurons):
                print(f"Neuron {neuron.name}:")
                print(f"Weights: {neuron.weights}")
                print(f"Bias: {neuron.bias}")


        # Print the weights' names of each neuron
        print("\nWeights' Names:")
        for layer in self.layers:
            for neuron in layer.neurons:
                for w in neuron.weights:
                    print(f"Weight: {w.name}, Value: {w.value}")

        print("\nDebugging complete.")


    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # def forward_propagation(self, input_data):
    #     layer_input = input_data
    #
    #     # Loop through each layer in the network
    #     for layer in self.layers:
    #         layer_output = []  # Store outputs of neurons in this layer
    #
    #         # Loop through each neuron in the layer
    #         for neuron in layer.neurons:
    #             weighted_sum = 0
    #
    #             # Only perform calculations for neurons in the input layer
    #             if neuron.layer_type == "input":
    #                 # Loop through the nested arrays in the input data
    #                 for data_array in layer_input:
    #                     # Ensure the lengths match before computing the dot product
    #                     # if len(data_array) == len(neuron.weights):
    #                     for i in range(len(data_array)-len(neuron.weights)):
    #                         weighted_sum += data_array[i] * neuron.weights[i].value + neuron.bias
    #                     # else:
    #                     #     # Handle the case where lengths don't match
    #                     #     print("Input data and weights have different lengths!")
    #
    #                 neuron_output = neuron.activation_function(weighted_sum)
    #                 layer_output.append(neuron_output)
    #
    #         # Update the input for the next layer with the current layer's output
    #         layer_input = np.array(layer_output)
    #     # Return the final output after passing through all layers
    #     return layer_input

    def forward_propagation(self, input_data):
        layer_input = input_data
        weighted_sum = 0
        # Loop through each layer in the network
        for layer in self.layers:
            layer_output = []  # Store outputs of neurons in this layer
            # Loop through each neuron in the layer
            for neuron in layer.neurons:
                # Only perform calculations for neurons in the input layer
                if neuron.layer_type == "input":
                    # Loop through the nested arrays in the input data
                    for data_array in layer_input:
                        for i in range(len(data_array)-len(neuron.weights)):
                            output = np.dot(data_array[i], neuron.weights[i].value) + neuron.bias
                            weighted_sum += output
                            print(f"weighted_sum: {weighted_sum}")
                            neuron_output = neuron.activation_function(weighted_sum)
                            layer_output.append(neuron_output)
                # elif neuron.layer_type == "hidden":
                #     # Loop through the nested arrays in the input data
                #     for data_array in layer_input:
                #         for i in range(len(data_array) - len(neuron.weights)):
                #             output = np.dot(data_array[i], neuron.weights[i].value) + neuron.bias
                #             weighted_sum += output
                #             print(f"weighted_sum: {weighted_sum}")
                #             neuron_output = neuron.activation_function(weighted_sum)
                #             layer_output.append(neuron_output)
        # Update the input for the next layer with the current layer's output
        layer_input = np.array(layer_input)
        # Return the final output after passing through all layers
        return layer_input
    # def forward_propagation(self, input_data):
    #     layer_input = input_data
    #
    #     # Loop through each layer in the network
    #     for layer in self.layers:
    #         layer_output = []  # Store outputs of neurons in this layer
    #
    #         # Loop through each neuron in the layer
    #         for neuron, data_array in layer.neurons, layer_input:
    #             weighted_sum = 0
    #
    #             # Only perform calculations for neurons in the input layer
    #             if neuron.layer_type == "input":
    #             # Loop through the nested arrays in the input data
    #                 for i in range(len(data_array)):
    #                     weighted_sum += data_array[i] * neuron.weights[i].value + neuron.bias
    #                 # Ensure the lengths match before computing the dot product
    #                 # if len(data_array) == len(neuron.weights):
    #                 weighted_sum += sum(x * w.value for x, w in zip(data_array, neuron.weights)) + neuron.bias
    #                 # else:
    #                     # Handle the case where lengths don't match
    #                 # print("Input data and weights have different lengths!")
    #
    #                 neuron_output = neuron.activation_function(weighted_sum)
    #                 layer_output.append(neuron_output)
    #
    #         # Update the input for the next layer with the current layer's output
    #         layer_input = np.array(layer_output)
    #
    #     # Return the final output after passing through all layers
    #     return layer_input
############################################################################################################
    # def forward_propagation_alaa(self, input_data):
    #     layer_input = input_data
    #
    #     # Loop through each layer in the network
    #     for layer in self.layers:
    #         layer_output = []  # Store outputs of neurons in this layer
    #
    #         # Loop through each neuron in the layer
    #         for neuron in layer.neurons:
    #             weighted_sum = 0
    #
    #             # Only perform calculations for neurons in the input layer
    #             if neuron.layer_type == "input":
    #                 # Print lengths outside of the loop for better understanding
    #                 print(f"neuron.weights length: {len(neuron.weights)}")
    #
    #                 # Loop through the nested arrays in the input data
    #                 for data_array in layer_input:
    #                     print(f"data_array length: {len(data_array)}")
    #
    #                     # Ensure the lengths match before computing the dot product
    #                     if len(data_array) == len(neuron.weights):
    #                         for i in range(len(data_array)):
    #                             weighted_sum += data_array[i] * neuron.weights[i].value + neuron.bias
    #                     else:
    #                         print("Input data and weights have different lengths!")
    #
    #                     neuron_output = neuron.activation_function(weighted_sum)
    #                     layer_output.append(neuron_output)
    #
    #         # Update the input for the next layer with the current layer's output
    #         layer_input = np.array(layer_output)
    #     # Return the final output after passing through all layers
    #     return layer_input
    # def forward_propagation_2(self, input_data):
    #             layer_input = input_data
    #
    #             # Loop through each layer in the network
    #             for layer in self.layers:
    #                 layer_output = []  # Store outputs of neurons in this layer
    #
    #                 # Loop through each neuron in the layer
    #
    #                 for neuron in layer.neurons:
    #
    #                     weight = neuron.weights
    #                     weighted_sum = np.dot(layer_input, [weight]) + neuron.bias
    #                     # Add bias
    #                     weighted_sum += neuron.bias
    #
    #                 # Update the input for the next layer with the current layer's output
    #                 layer_input = np.array(layer_output)
    #
    #             # Return the final output after passing through all layers
    #             return layer_input

    # def initialize_weights(self):
    #     # Initialize weights randomly for all neurons in all layers
    #     for layer in self.layers:
    #         for neuron in layer.neurons:
    #             neuron.weights = np.random.rand(len(neuron.weights))

    # def forward_propagation(self, input_data):
    #     # Step 2: Loop over layers
    #     for layer in self.layers:
    #         layer_outputs = []
    #
    #         # Step 3: Loop over neurons in the layer
    #         for neuron in layer.neurons:
    #             # Step 4: Calculate and store output
    #             weighted_sum = np.dot(input_data, [weight.value for weight in neuron.weights]) + neuron.bias
    #             neuron.output = neuron.activation_function(weighted_sum)
    #             layer_outputs.append(neuron.output)
    #
    #         input_data = np.array(layer_outputs)  # Set the input for the next layer
    #
    #     return input_data  # Return the final output of the neural network

    # def forward_propagation(self, input_data):
    #     # Step 2: Loop over layers
    #     for layer in self.layers:
    #         layer_outputs = []
    #
    #         # Step 3: Loop over neurons in the layer
    #         for neuron in layer.neurons:
    #             weighted_sums = []
    #
    #             # Step 4: Calculate weighted sum for each element in input_data
    #             for data_array in input_data:
    #                 if np.isscalar(data_array):
    #                     # If data_array is a scalar, convert it to a 1-element array
    #                     data_array = np.array([data_array])
    #
    #                 # Check if data_array is an iterable (array-like)
    #                 if hasattr(data_array, "__iter__") or isinstance(data_array, np.ndarray):
    #                     weighted_sum = np.dot(data_array, [weight.value for weight in neuron.weights]) + neuron.bias
    #                     weighted_sums.append(weighted_sum)
    #                 else:
    #                     # Handle the case where data_array is a single value (scalar)
    #                     weighted_sum = data_array * neuron.weights[0].value + neuron.bias
    #                     weighted_sums.append(weighted_sum)
    #
    #             # Calculate the final weighted sum by summing the weighted sums from each data_array
    #             final_weighted_sum = sum(weighted_sums)
    #
    #             # Apply activation function to the final weighted sum
    #             neuron.output = neuron.activation_function(final_weighted_sum)
    #             layer_outputs.append(neuron.output)
    #
    #         input_data = np.array(layer_outputs)  # Set the input for the next layer
    #
    #     return input_data  # Return the final output of the neural network
    #
    # # def forward_propagation(self, input_data):## original fun
    # #     # Step 2: Loop over layers
    # #     for layer in self.layers:
    # #         layer_outputs = []
    # #
    # #         # Step 3: Loop over neurons in the layer
    # #         for neuron in layer.neurons:
    # #             # Step 4: Calculate and store output
    # #             neuron_output = neuron.calculate_output(input_data)
    # #             layer_outputs.append(neuron_output)
    # #
    # #         input_data = np.array(layer_outputs)  # Set the input for the next layer
    # #
    # #     return input_data  # Return the final output of the neural network
    #
    def backward_propagation(self, targets, input_data):
        # Step 7: Loop over output neurons
        for i, output_neuron in enumerate(self.layers[-1].neurons):
            # Step 8: Calculate and store δko
            output_neuron.delta = (output_neuron.output - targets[i]) * output_neuron.calculate_derivative()

        # Step 9: Loop over hidden neurons
        for hidden_neuron in self.layers[1].neurons:
            # Step 10: Calculate and store δjh
            hidden_neuron.delta = np.dot(
                [neuron.weights[i] * neuron.delta for i, neuron in enumerate(self.layers[-1].neurons)],
                hidden_neuron.calculate_derivative())

        # Step 11-14: Update weights
        for i, output_neuron in enumerate(self.layers[-1].neurons):
            for j, hidden_neuron in enumerate(self.layers[1].neurons):
                # Step 11: Update weight
                output_neuron.weights[j] -= self.learning_rate * output_neuron.delta * hidden_neuron.output

        for i, hidden_neuron in enumerate(self.layers[1].neurons):
            for j, input_feature in enumerate(input_data):
                # Step 12: Update weight
                hidden_neuron.weights[
                    j] -= self.learning_rate * hidden_neuron.delta * hidden_neuron.calculate_derivative() * input_feature
    def train(self, features, targets):
        print("Step 3_B: Training the neural network...")

        # # Step 1: Initialize weights randomly
        # self.initialize_weights()

        for epoch in range(self.num_epochs):
            mean_squared_error = 0

            for i in range(len(features)):
                # Step 2a: Forward propagation
                predictions = self.forward_propagation(features[i])

                # Step 2b: Backward propagation
                self.backward_propagation(targets[i], features[i])

                # Calculate mean squared error for this example
                mean_squared_error += self.mean_squared_error(predictions, targets[i])

            # Calculate mean squared error for the epoch
            mean_squared_error /= len(features)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Mean Squared Error: {mean_squared_error}")
    #
    # def test(self, test_features, test_targets):
    #     print("Testing the trained neural network...")
    #
    #     overall_error = 0
    #     predictions = []
    #
    #     for i in range(len(test_features)):
    #         # Step 3a: Forward propagation for testing
    #         prediction = self.predict(test_features[i])
    #         predictions.append(prediction)
    #
    #         # Step 3b: Calculate the error
    #         example_error = self.mean_squared_error(prediction, test_targets[i])
    #         overall_error += example_error
    #
    #         # Print individual example error if needed
    #         print(f"Example {i + 1} Error: {example_error}")
    #
    #     # Calculate overall accuracy or error for the test set
    #     overall_error /= len(test_features)
    #     print(f"Overall Mean Squared Error on Test Set: {overall_error}")
    #
    #     return predictions  # Return predictions for accuracy calculation
    #
    # def accuracy(self, predictions, targets, tolerance=0.1):
    #     # Calculate the percentage of predictions within the tolerance
    #     correct_predictions = np.sum(np.abs(predictions - targets) <= tolerance)
    #     total_examples = len(targets)
    #
    #     accuracy = correct_predictions / total_examples
    #     return accuracy
    #
    # def predict(self, new_data):
    #     # Step 2: Loop over layers
    #     for layer in self.layers:
    #         layer_outputs = []
    #
    #         # Step 3: Loop over neurons in the layer
    #         for neuron in layer.neurons:
    #             # Step 4: Calculate and store output
    #             neuron_output = neuron.calculate_output(new_data)
    #             layer_outputs.append(neuron_output)
    #
    #         new_data = np.array(layer_outputs)  # Set the input for the next layer
    #
    #     return new_data  # Return the final output of the neural network for prediction
    #
    def mean_squared_error(self, predictions, targets):#1/2 sumation
        # Calculate the mean squared error
        return np.mean((predictions - targets) ** 2)
