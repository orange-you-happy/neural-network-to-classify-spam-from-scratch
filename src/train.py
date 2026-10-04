import numpy as np
from preprocess import load_data
from model import gradient_descent, forward_prop, get_predictions, get_accuracy

N_HIDDEN = 300
ALPHA = 1
ITERATIONS = 1000

if __name__ == "__main__":
    np.random.seed(0)
    X_train, X_test, y_train, y_test, vectoriser = load_data()
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T
    y_test = y_test.T
    print("Training: ")
    W1, W2, b1, b2 = gradient_descent(X_train, y_train, ALPHA, ITERATIONS, N_HIDDEN)
    print("Running on test data:")
    _, _, _, A2_test = forward_prop(W1, W2, b1, b2, X_test)
    test_predictions = get_predictions(A2_test)
    print(f"Test accuracy: {get_accuracy(test_predictions, y_test):.2f}")
