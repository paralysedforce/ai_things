import numpy as np

batch_size = 10 
bptt_cutoff = 3

class RNN(object):
    def __init__(self, input_size, state_size, output_size):
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        self.U = self._initialize_weights(self.input_size, self.state_size)
        self.W = self._initialize_weights(self.state_size, self.state_size)
        self.V = self._initialize_weights(self.state_size, self.output_size)

        self.state = np.zeros(self.state_size)
        self.prev_states = []

    def _initialize_weights(self, size_in, size_out):
        return np.random.uniform(-1 / np.sqrt(size_in), 1 / np.sqrt(size_in), (size_out, size_in))

    def forward(self, x, training=False):
        if len(x) != self.input_size:
            raise ValueError("Input of incorrect dimension")

        self.state = np.tanh( self.U @ x + self.W @ self.state )
        if training:
            self.prev_states.append(self._state)
        return softmax(V @ self.state)

    def train(self, training_data, training_labels):
        # Gradients for the weight matrices- change in loss L with regard to weight
        self.dLdU = np.zeros(self.U.shape)
        self.dLdW = np.zeros(self.W.shape)
        self.dLdV = np.zeros(self.V.shape)

        for x, y in zip(training_data, training_labels):
            for state in self.prev_states:
                self._bptt_step(x, y, output, state)

            while len(self.prev_states) >= bptt_cutoff:
                self.prev_states = self.prev_states.remove(0)
        
        self.U += self.dLdU
        self.W += self.dLdW
        self.V += self.dLdW

    def _bptt_step(self, x, y, output):
        yHat = self.forward(x, training=True)
        dLdyHat = loss(yHat, y, deriv=True)

        for state in reversed(self.states):



    def predict(self, x):
        # For classification
        output = self.forward(x)
        return np.argmax(output)

    def evaluate_test_data(self, test_data, test_labels):
        correct_predictions = 0
        for x, label in zip(test_data, test_labels):
            y = self.predict(x)
            if y == label:
                correct_predictions += 1
        return correct_predictions

# Helpers #
def softmax(x, deriv=False):
    if not deriv:
        v = np.exp(x)
        return v / np.sum(v)
    else:
        s = softmax(x)
        return s * (1 - s )

def loss(predicted, actual, deriv=False):
    if not deriv:
        return -actual * np.log(predicted)
    else:
        return -actual / predicted

def batch(dataset, size=batch_size):
    return np.array([dataset[i: i+size] for i in range(0, len(dataset) - size, size)])

def batch_loss(predicted, actual):
    num_examples = len(actual)
    total_loss = sum(loss(p, a) for e, a in zip(predicted, actual))
    return total_loss / num_examples
