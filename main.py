import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) ##
        self.B1 = np.random.randn(hidden_size, 1)
        self.W2 = np.random.randn(output_size, hidden_size)
        self.B2 = np.random.randn(output_size, 1)
    
    def forward(self, x):
        self.Z1 = np.dot(self.W1, x) + self.B1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = softmax(self.Z2)

        self.X = x

        return self.A2

    def compCost(self, Y, Y_hat):
        m = Y.shape[1]
        cost = np.sum((Y - Y_hat) ** 2) / m
        return cost
    
    def backwards(self, Y, Y_hat):
        self.dZ2 = Y_hat - Y
        m = Y.shape[1]

        self.dW2 = (1/m) * np.dot(self.dZ2, self.A1.T)
        self.dB2 = (1/m) * np.sum(self.dZ2, axis=1, keepdims=True)

        self.dA1 = np.dot(self.W2.T, self.dZ2)
        self.dZ1 = self.dA1 * self.A1 * (1 - self.A1)
        self.dW1 = (1/m) * np.dot(self.dZ1, self.X.T)
        self.dB1 = (1/m) * np.sum(self.dZ1, axis=1, keepdims=True)

    def update_parameters(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.B1 -= learning_rate * self.dB1
        self.W2 -= learning_rate * self.dW2
        self.B2 -= learning_rate * self.dB2





    
# Example inputs
X = np.random.randn(3, 5)
Y = np.zeros((26, 5))
Y[0, 0] = 1  # A
Y[2, 1] = 1  # C
Y[25, 2] = 1 # Z
Y[1, 3] = 1  # B
Y[4, 4] = 1  # E

# Create a network, Input with 3 nodes, Hidden layer with 4 nodes, output with 1 node.
nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=26)
costs = []
epochs = 100
learning_rate = 0.01

for epoch in range(epochs):
    Y_hat = nn.forward(X)
    cost = nn.compCost(Y, Y_hat)
    costs.append(cost)

    nn.backwards(Y, Y_hat)
    nn.update_parameters(learning_rate)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Cost = {cost:.4f}")



plt.figure(figsize=(8, 5))
plt.plot(costs, label="Cost")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()



# Final predictions after training
Y_hat = nn.forward(X)
true_labels = np.argmax(Y, axis=0)
predicted_labels = np.argmax(Y_hat, axis=0)

label_map = [chr(i) for i in range(65, 91)] 

plt.figure(figsize=(10, 6))
x = np.arange(X.shape[1])
width = 0.35

plt.bar(x - width/2, true_labels, width, label='True', color='red')
plt.bar(x + width/2, predicted_labels, width, label='Predicted', color='blue')

plt.xticks(x, [f"Sample {i}" for i in range(X.shape[1])])
plt.yticks(ticks=np.arange(26), labels=label_map)
plt.ylabel("Class")
plt.title("True vs. Predicted Labels (After Training)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("predictions_after_training.png", dpi=300)
plt.show()
