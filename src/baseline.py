from preprocess import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectoriser = load_data()

    #LogisticRegression works by finding a weighted sum for each possible answer, then squashing it down to a probability between 0 and 1. 
    #It then chooses the one with the highest probability. 
    model = LogisticRegression(max_iter=200)

    #fit actually trains the model
    model.fit(X_train, y_train.ravel())
    
    #predict runs trained model on data unseen
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test.ravel(), predictions)
    #Confusion matrix analyses where it went wrong
    confusion = confusion_matrix(y_test.ravel(), predictions)
    print(f"Accuracy: {accuracy:.2f}") 
    print(confusion)
    