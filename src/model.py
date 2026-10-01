import numpy as np

def init_params(n_input, n_hidden, n_classes): #Initial values for the model
    #Wn = Weights for hidden layer n, is a matrix since each neuron has a connection to each neuron of the next layer
    #n_hidden is the number of neurons each hidden layer, n_input is the number of inputs
    #This creates a n_hidden x n_input matrix, we subtract 0.5 to centre it at 0
    W1 = np.random.rand(n_hidden, n_input) - 0.5
    W2 = np.random.rand(n_classes, n_hidden) - 0.5

    #bn = Biases for each neuron in layer n
    b1 = np.random.rand(n_hidden, 1) - 0.5
    b2 = np.random.rand(n_classes, 1) - 0.5

    return W1, W2, b1, b2

def ReLU(Z): #One of the activation functions used
    #This applies max(0,Z) for every number in the matrix
    return np.maximum(0, Z)


def sigmoid(Z): #One of the activation functions used, turns every number in between 0 and 1.
    Z = np.clip(Z, -500, 500)        
    return 1 / (1 + np.exp(-Z))

def forward_prop(W1, W2, b1, b2, X): #The process of actually making a prediction
    #Zn = the hidden layer's weighted sum + the bias before the activation function
    #An is the output after the activation function
    #We do this with matrix multiplication (@ is the operator), we multiply the matrices to get the weighted sum

    Z1 = W1 @ X + b1 #This matrix has dimensions of W1 columns x b1 rows
    A1 = ReLU(Z1)

    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    return Z1, Z2, A1, A2

def deriv_ReLU(Z):
    #When the number > 0, the derivative is 1, otherwise it's 0 (booleans converted to integers during calculation)
    return Z > 0

def backward_prop(Z1, A1, A2, W2, X, Y): #Find how much the parameters should change after prediction
    m = Y.size

    #dZn = How much Zn should change to reduce the loss (error)
    dZ2 = A2 - Y
    dZ1 = W2.T @ dZ2 * deriv_ReLU(Z1)

    #dWn = How much Wn should change to reduce the loss
    dW2 = 1 / m * dZ2 @ A1.T
    dW1 = 1 / m * dZ1 @ X.T

    #dBn = How much Bn should change to reduce the loss
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, dW2, db1, db2

def update_params(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha):
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    return W1, W2, b1, b2

def get_predictions(A2):
    return (A2 > 0.5).astype(int)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def gradient_descent(X, Y, alpha, iterations, n_hidden): #Actually updates the parameters
    n_input = X.shape[0]
    W1, W2, b1, b2 = init_params(n_input, n_hidden, 1)
    #For each iteration, predict, move back and find how much to change.
    for i in range(iterations):
        Z1, Z2, A1, A2 = forward_prop(W1, W2, b1, b2, X)
        dW1, dW2, db1, db2 = backward_prop(Z1, A1, A2, W2, X, Y)
        W1, W2, b1, b2 = update_params(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha)

        if i % 50 == 0:
            print("Iteration:", i)
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y):.2f}")

    print("Iteration:", i)
    print(f"Accuracy: {get_accuracy(get_predictions(A2), Y):.2f}")

    return W1, W2, b1, b2





    


