This is a neural network that I wrote from scatch in python using numpy. It utilises forward and backward propagation to adjust neurons represented as matrices. The activation functions used here are ReLU and sigmoid. Sigmoid is used because we only have 2 results: Either it's a spam or not a spam.
Training and testing dataset: https://huggingface.co/datasets/SetFit/enron_spam

Setup:
1. git clone https://github.com/orange-you-happy/neural-network-to-classify-spam-from-scratch.git
2. cd neural-network-to-classify-spam-from-scratch
3. Make a venv, then pip install -r requirements.txt
4. To run a baseline logistic regression: python src/baseline.py
5. To run the actual neural network: python src/train.py
