import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
# Features = fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id
data = pd.read_csv('WineQT.csv')


# Convert the numerical target variable into categorical
bins = [0, 4, 7, 10]  # Define the range of values for each category
labels = ['bad', 'average', 'good']  # Define the labels for each category
data['quality'] = pd.cut(data['quality'], bins=bins, labels=labels)  # Convert the 'quality' column to categorical data

# Encode the categorical target variable
le = LabelEncoder()  # Initialize a LabelEncoder object
data['quality'] = le.fit_transform(data['quality'])  # Encode the 'quality' column with numerical labels

# Split the dataset into training and testing sets
X = data.drop('quality', axis=1).values
y = data['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the sigmoid function
def sigmoid(x):
    # Clip the input to the exp function to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
# used during the backpropagation process
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases for the hidden layer(with 4 neurons) and the output layer(with 1 neuron)
weights0 = 2 * np.random.random((X_train.shape[1], 4)) - 1  # weights for hidden layer(with 4 neurons)
# Multiplying by 2 and subtracting 1 at the end transforms the range of these random numbers from [0, 1) to [-1, 1).
weights1 = 2 * np.random.random((4, 1)) - 1  # weights for output layer(with 1 neuron)
bias0 =  1 # bias for hidden layer
bias1 = 1  # bias for output layer

epochs = 10

# Train the MLP
for i in range(epochs):
    # Forward pass
    layer0 = X_train # Input layer
    layer1 = sigmoid(np.dot(layer0, weights0) + bias0) # Hidden layer
    layer2 = sigmoid(np.dot(layer1, weights1) + bias1) # Output layer

    # Backward pass
    layer2_error = layer2 - y_train.reshape(-1, 1)
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    layer1_error = np.dot(layer2_delta, weights1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    learning_rate = .25  # Define your learning rate

    # Update weights and biases
    weights1 -= learning_rate * np.dot(layer1.T, layer2_delta)
    bias1 -= learning_rate * np.sum(layer2_delta, axis=0)

    weights0 -= learning_rate * np.dot(layer0.T, layer1_delta)
    bias0 -= learning_rate * np.sum(layer1_delta, axis=0)

# Predict the test set results
layer1 = sigmoid(np.dot(X_test, weights0) + bias0)
y_pred = sigmoid(np.dot(layer1, weights1) + bias1)
y_pred = np.round(y_pred)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)