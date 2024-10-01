import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, seed=10):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.loss_over_time = []
 
        np.random.seed(seed)
        # Initialize weights with He initialization suited for ReLU
        self.weights_input_hidden = np.random.randn(self.input_nodes, self.hidden_nodes) * np.sqrt(2. / self.input_nodes)
        self.weights_hidden_output = np.random.randn(self.hidden_nodes, self.output_nodes) * np.sqrt(2. / self.hidden_nodes)

        # Biases can be initialized to zeros
        self.bias_hidden = np.zeros(self.hidden_nodes)
        self.bias_output = np.zeros(self.output_nodes)
    
    def forward_pass(self, inputs):
        self.inputs = inputs
        self.hidden_layer_activation = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = relu(self.hidden_layer_activation)
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = relu(self.output_layer_activation)  # Consider if ReLU is appropriate for the output layer
        
    def backward_pass(self, actual_output):
        loss = np.mean(np.square(actual_output - self.predicted_output))  # Mean squared error loss
        self.loss_over_time.append(loss)

        d_loss_predicted_output = -2 * (actual_output - self.predicted_output)
        d_predicted_output = d_loss_predicted_output * relu_derivative(self.predicted_output)
        
        error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * relu_derivative(self.hidden_layer_output)
        
        # Update the weights and biases with learning rate
        learning_rate = 0.0015
        self.weights_hidden_output -= learning_rate * self.hidden_layer_output.T.dot(d_predicted_output)
        self.weights_input_hidden -= learning_rate * self.inputs.T.dot(d_hidden_layer)
        self.bias_output -= learning_rate * np.sum(d_predicted_output, axis=0)
        self.bias_hidden -= learning_rate * np.sum(d_hidden_layer, axis=0)

    def train(self, inputs, actual_output):
        self.forward_pass(inputs)
        self.backward_pass(actual_output)

def train_network_with_user_input():
    file_path = 'log_standardised_data.xlsx'
    X = pd.read_excel(file_path, nrows=316)  
    inputs = X.iloc[:, 0:7].values 
    outputs = X.iloc[:, 7].values

    validationFile = pd.read_excel(file_path, nrows = 106, skiprows = 316)
    validationSet = validationFile.iloc[:, 0:7].values
    validationOutput = validationFile.iloc[:, 7].values


    input_nodes = 7
    output_nodes = 1
    epochs = 10000

    while True:
        try:
            hidden_nodes = int(input("Enter the number of hidden nodes (or type 'exit' to quit): "))
        except ValueError:
            print("Exiting program.")
            break

        print(f"\nTraining with {hidden_nodes} hidden nodes.")
        floodIndexNetwork = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        
        for epoch in range(epochs):
            floodIndexNetwork.train(inputs, outputs.reshape(-1, 1))
            if epoch == 9999:
                predicted_outputs = []
                actual_outputs = []
                i = 0
                while i < 100:
                    floodIndexNetwork.forward_pass(validationSet[i])  
                    predicted_outputs.append(floodIndexNetwork.predicted_output)
                    actual_outputs.append(validationOutput[i])
                    print(floodIndexNetwork.predicted_output)  
                    print(validationOutput[i])
                    print("")
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
        plt.title('Validation Set Testing')
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

# Call the function to start the training process based on user input
train_network_with_user_input() 