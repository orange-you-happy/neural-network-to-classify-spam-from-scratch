import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_file(path):
    df = pd.read_json(path, lines=True)
    X = df["text"].fillna("")
    y = df["label"].to_numpy().reshape(-1, 1).astype(np.float32)

    return X, y


def load_data():
    X_train, y_train = load_file("data/train.jsonl")
    X_test, y_test  = load_file("data/test.jsonl")

    vectoriser = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectoriser.fit_transform(X_train)   
    X_test  = vectoriser.transform(X_test)        

    return X_train, X_test, y_train, y_test, vectoriser

