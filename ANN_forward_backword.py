import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        
        # Initialize weights and biases with random values
        self.hidden_weights = np.random.rand(input_neurons, hidden_neurons)
        self.hidden_bias = np.random.rand(1, hidden_neurons)
        self.output_weights = np.random.rand(hidden_neurons, output_neurons)
        self.output_bias = np.random.rand(1, output_neurons)
    
    # Forward propagation
    def forward_propagation(self, X):
        self.hidden_activation = sigmoid(np.dot(X, self.hidden_weights) + self.hidden_bias)
        self.output_activation = sigmoid(np.dot(self.hidden_activation, self.output_weights) + self.output_bias)
    
    # Backpropagation
    def backpropagation(self, X, y, learning_rate):
        # Compute output layer error
        output_error = y - self.output_activation
        output_delta = output_error * sigmoid_derivative(self.output_activation)
        
        # Compute hidden layer error
        hidden_error = output_delta.dot(self.output_weights.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_activation)
        
        # Update weights and biases
        self.output_weights += self.hidden_activation.T.dot(output_delta) * learning_rate
        self.output_bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.hidden_weights += X.T.dot(hidden_delta) * learning_rate
        self.hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    
    # Training function
    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            self.forward_propagation(X)
            self.backpropagation(X, y, learning_rate)
    
    # Prediction function
    def predict(self, X):
        self.forward_propagation(X)
        return self.output_activation

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize the neural network
nn = NeuralNetwork(input_neurons=2, hidden_neurons=4, output_neurons=1)

# Train the neural network
nn.train(X, y, learning_rate=0.1, epochs=10000)

# Make predictions
predictions = nn.predict(X)
print("Predictions:")
print(predictions)
