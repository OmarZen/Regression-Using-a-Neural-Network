# Regression Using a Neural Network

## About the problem:
Regression is a supervised learning problem that aims at estimating the relationships between a dependent variable (target) and one or more independent variables (features). Artificial neural networks are widely used in regression as they can learn the complex non-linear relationship between the features and target.

## What you are required to do:
Implement a feedforward neural network (from scratch) that predicts the cement strength. This FFNN should have 3 layers: input, hidden, and output.
- Use only ONE hidden layer.
- Use Sigmoid activation function for the hidden layer.

You will be given the “concrete_data.xlsx” file which contains 700 records of concrete construction data. Each record is composed of 5 columns representing the cement, water, superplasticizer, age, and the target to be predicted which is concrete strength.

## Important remarks to help you solve the problem:
1. Load the data from the file into the appropriate structures. The features and targets should be in 2 separate arrays because the neural network only uses the features as its input in the forward propagation step, while the targets are used as the actual outputs in the backpropagation step. The length of each of these 2 arrays should be equal to the number of data records. Each entry in the features array should also be an array storing the numbers in the first 4 columns of a record.
2. You may need to normalize the data (optional step to get better results).
3. Split the data into training (75% of the data) and testing (25% of the data) sets.
4. Implement the class “NeuralNetwork”. This class can have any needed attributes and it must have methods to:
   a. Set the architecture and hyperparameters of the NN such as the number of neurons in each layer, the number of epochs, etc.
   b. Train the NN for a number of epochs by performing forward and backward propagation on each training example (each data record in the training set).
   c. Predict the target value of new data if the user chooses to enter a new data record to get the cement strength. You can do this by performing forward propagation using the final weights. (Note: This method could take one or more training examples)
   d. Calculate the error of the NN model.

Note: You can implement any classes you want (Layer, Neuron, etc.) to be used by the “NeuralNetwork” class.
