import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:

    ## initialising class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, seed=10):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.loss_over_time = []

        np.random.seed(seed)  # Ensure consistency within testing
        # Initialize weights and biases with Xavier/Glorot initialization
        self.weights_input_hidden = np.random.randn(self.input_nodes, self.hidden_nodes) * np.sqrt(2. / (self.input_nodes + self.hidden_nodes))
        self.weights_hidden_output = np.random.randn(self.hidden_nodes, self.output_nodes) * np.sqrt(2. / (self.hidden_nodes + self.output_nodes))
        self.bias_hidden = np.zeros(self.hidden_nodes)
        self.bias_output = np.zeros(self.output_nodes)

        # Momentum
        self.velocity_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.velocity_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.velocity_bias_hidden = np.zeros_like(self.bias_hidden)
        self.velocity_bias_output = np.zeros_like(self.bias_output)
        self.momentum = 0.9

        # Bold Driver 
        self.learning_rate = 0.0015
        self.lr_increase = 1.05  
        self.lr_decrease = 0.7   
        self.prev_loss = float('inf')  

    # forward pass of the network - uses weights and biases and transfer function
    def forward_pass(self, inputs):
        self.inputs = inputs
        self.hidden_layer_activation = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = sigmoid(self.output_layer_activation)
        
    # backward pass of the network
    def backward_pass(self, actual_output):

        # calculating loss for analytical use
        loss = np.mean(np.square(actual_output - self.predicted_output))
        self.loss_over_time.append(loss)


        # Adjust learning rate with Bold Driver
        if loss < self.prev_loss:
            self.learning_rate *= self.lr_increase
        else:
            self.learning_rate *= self.lr_decrease
        self.prev_loss = loss

        d_loss_predicted_output = -2 * (actual_output - self.predicted_output)
        d_predicted_output = d_loss_predicted_output * sigmoid_derivative(self.predicted_output)
        
        error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_output)
        
        # Calculate updates with momentum and adjusted learning rate
        self.velocity_weights_hidden_output = (self.momentum * self.velocity_weights_hidden_output - 
                                               self.learning_rate * self.hidden_layer_output.T.dot(d_predicted_output))
        self.velocity_weights_input_hidden = (self.momentum * self.velocity_weights_input_hidden - 
                                              self.learning_rate * self.inputs.T.dot(d_hidden_layer))
        self.velocity_bias_output = (self.momentum * self.velocity_bias_output - 
                                     self.learning_rate * np.sum(d_predicted_output, axis=0))
        self.velocity_bias_hidden = (self.momentum * self.velocity_bias_hidden - 
                                     self.learning_rate * np.sum(d_hidden_layer, axis=0))

        # Update weights and biases
        self.weights_hidden_output += self.velocity_weights_hidden_output
        self.weights_input_hidden += self.velocity_weights_input_hidden
        self.bias_output += self.velocity_bias_output
        self.bias_hidden += self.velocity_bias_hidden
        
    def train(self, inputs, actual_output):
        # training by passing all data through the network forward and back
        self.forward_pass(inputs)
        self.backward_pass(actual_output)


# interface
def train_network_with_user_input():
    # data splitting
    file_path = 'log_standardised_data.xlsx'
    X = pd.read_excel(file_path, nrows=316)  
    inputs = X.iloc[:, 0:7].values 
    outputs = X.iloc[:, 7].values


    validationFile = pd.read_excel(file_path, nrows = 106, skiprows = 316)
    validationSet = validationFile.iloc[:, 0:7].values
    validationOutput = validationFile.iloc[:, 7].values

    testingFile = pd.read_excel(file_path, nrows = 105, skiprows = 422)
    testingSet = validationFile.iloc[:, 0:7].values
    testingOutput = validationFile.iloc[:, 7].values

    input_nodes = 7 # for fixed problem
    output_nodes = 1 # for fixed problem 
    epochs = 10000


        
    hidden_nodes = int(input("Enter the number of hidden nodes: "))
    # Input 8 for specific problem (flood index) 

    print(f"\nTraining with {hidden_nodes} hidden nodes.")
    floodIndexNetwork = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        
    for epoch in range(epochs):
        floodIndexNetwork.train(inputs, outputs.reshape(-1, 1))
        if epoch == 9999:
            predicted_outputs = []
            actual_outputs = []
            i = 0
            squared_errors = []

            # collecting loss and prediction data for analytical purposes
            while i < 106:
                floodIndexNetwork.forward_pass(testingSet[i])  # using testing set
                predicted_outputs.append(floodIndexNetwork.predicted_output)
                actual_outputs.append(testingOutput[i])
                squared_error = (floodIndexNetwork.predicted_output - validationOutput[i])**2
                squared_errors.append(squared_error)
                i = i + 1

    # loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(floodIndexNetwork.loss_over_time, label='Loss over epochs')
    plt.title('Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(0, 0.2)
    plt.yticks(np.arange(0, 0.2, step=0.02))
    plt.show()

    # predicted vs actual graph
    plt.figure(figsize=(10, 5))
    plt.scatter(actual_outputs, predicted_outputs, color='blue')
    plt.title('Testing Set Testing')
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    # line y = x 
    min_val = min(min(predicted_outputs), min(actual_outputs))
    max_val = max(max(predicted_outputs), max(actual_outputs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

    # metrics

    predicted_outputs = np.array(predicted_outputs).flatten()
    actual_outputs = np.array(actual_outputs).flatten()

    mse = np.mean(squared_errors)
    print(f"Mean Squared Error (MSE) on Testing Set: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE) on Testing Set: {rmse}")
    mae = np.mean(np.abs(actual_outputs - predicted_outputs))
    print(f"Mean Absolute Error (MAE) on Testing Set: {mae}")
    # coefficient of determination
    ss_res = np.sum((actual_outputs - predicted_outputs) ** 2)
    ss_tot = np.sum((actual_outputs - np.mean(actual_outputs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared (R^2): {r_squared}")

while True: 
    train_network_with_user_input() 