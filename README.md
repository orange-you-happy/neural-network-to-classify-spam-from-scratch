Note: If my CV said 92%, it's because the parameters were N_HIDDEN = 50, ALPHA = 0.2, ITERATIONS = 500, which is relatively small and is able to give a result. Rerunning with the current variables set will give 98%.

This is a neural network that I wrote from scratch in python using numpy. It utilises forward and backward propagation to adjust neurons represented as matrices. The activation functions used here are ReLU and sigmoid. Sigmoid is used because we only have 2 results: Either it's a spam or not a spam.
Training and testing dataset: https://huggingface.co/datasets/SetFit/enron_spam

Setup:
1. git clone https://github.com/orange-you-happy/neural-network-to-classify-spam-from-scratch.git
2. cd neural-network-to-classify-spam-from-scratch
3. Make a venv, then pip install -r requirements.txt
4. To run a baseline logistic regression: python src/baseline.py
5. To run the actual neural network: python src/train.py
6. To adjust the parameters, edit the constants in train.py.

Results:
- The trained neural network is capable of detecting spam 98% of the time, though the baseline is capable of up to 99%. We could probably improve it even further if I continue to up the number of neurons and learning rate
- I intend to soon attempt to use this trained AI model and integrate this with an email sorter
- <img width="1271" height="902" alt="image" src="https://github.com/user-attachments/assets/a60644e5-8ae5-458f-a0e0-745f562d9639" />
