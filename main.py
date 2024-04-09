import pandas as pd
from NeuralNetwork import NeuralNetwork
import numpy as np

# Step 1: Load Data and Perform Preprocessing

# Step 1: Load Data and Perform Preprocessing
def load_and_preprocess_data(file_path):
    print("Step 1: Loading data and performing preprocessing...")
    data = pd.read_excel(file_path)

    # Extract features and targets
    features = data.iloc[:, :-1].values  # Select all columns except the last one
    targets = data.iloc[:, -1].values    # Select the last column

    # Normalize features
    normalized_features = normalize(features)

    return normalized_features, targets
# Normalization function
def normalize(data):
    # Normalize the data using the given formula
    normalized_data = data / np.sum(data, axis=1, keepdims=True)
    return normalized_data

# Step 2: Split Data into Training and Testing Sets
def split_data(features, targets):
    print("Step 2: Splitting data into training and testing sets...")
    features_train = features[:int(0.75 * len(features))]
    targets_train = targets[:int(0.75 * len(targets))]
    features_test = features[int(0.75 * len(features)):]
    targets_test = targets[int(0.75 * len(targets)):]
    return features_train, targets_train, features_test, targets_test


# Step 3: Main Function
def main():
    # Set the file path
    file_path = "concrete_data.xlsx"

    # Step 1: Load Data and Perform Preprocessing
    features, targets = load_and_preprocess_data(file_path)

    # Initialize Neural Network
    nn = NeuralNetwork()
    # Step 2: Split Data into Training and Testing Sets (75-25 split)
    features_train, targets_train, features_test, targets_test = split_data(features, targets)

    # Set Neural Network Architecture and Hyperparameters
    nn.set_architecture(input_size=features_train.shape[1], hidden_size=2, output_size=1, learning_rate=0.5, num_epochs=100)

    # Step 3: Train the Neural Network
    nn.train(features_train, targets_train)

    # Step 4: Test the Neural Network and Capture Predictions
    predictions = nn.test(features_test, targets_test)

    # Step 5: Calculate and Print Accuracy
    accuracy = nn.accuracy(predictions, targets_test)
    print(f"Overall Accuracy(or error) on Test Set: {accuracy * 100:.2f}%")

    # Step 5: Use the Trained Model for Prediction
    new_data_point = features_test[0]  # Example: Take the first data point from the test set
    prediction = nn.predict(new_data_point)
    print(f"Prediction for new data point: {prediction}")

if __name__ == "__main__":
    main()
