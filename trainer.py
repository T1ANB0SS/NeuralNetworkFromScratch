import time

import numpy as np
from matplotlib import pyplot as plt
import neural_network as nn
import utils

# AI CONFIGURATION
LEARN_RATE = 0.05
BATCH_SIZE = 200
MOMENTUM = 0.9
EPOCHS = 20

AI = nn.NeuralNetwork((utils.INPUT_SIZE, 100, 32, 10))
AI.randomize()

# 'one-hot encoding' encodes labels to a table of expected outputs. Use np.eye
x_train = utils.x_train.reshape(utils.x_train.shape[0], -1)
y_train = np.eye(10, dtype=np.float32)[utils.y_train]

x_train_aug = utils.x_train_aug.reshape(utils.x_train_aug.shape[0], -1)
y_train_aug = np.eye(10, dtype=np.float32)[utils.y_train_aug]

x_test = utils.x_test.reshape(utils.x_test.shape[0], -1)
y_test = np.eye(10, dtype=np.float32)[utils.y_test]

### SHOW EXAMPLES ###
utils.show_examples()


### TRAIN THE AI ###
index = 0
epoch = 0

plt.ion()

fig, ax = plt.subplots()

epochs = []
test_accuracies = []
train_accuracies = []

line_test, = ax.plot([], [], label="Test")
line_train, = ax.plot([], [], label="Train")

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training Progress")
ax.legend()

def show_accuracy():
    test_accuracy = AI.accuracy(x_test[:1000], utils.y_test[:1000]) * 100
    train_accuracy = AI.accuracy(x_train, utils.y_train) * 100

    print(f"\nTest Accuracy: {test_accuracy : .4f}%")
    print(f"Training Accuracy: {train_accuracy : .4f}%")
    print(f"Epoch: {epoch:.4f}")

    # Add data
    epochs.append(epoch)
    test_accuracies.append(test_accuracy)
    train_accuracies.append(train_accuracy)

    # Update graph
    line_test.set_data(epochs, test_accuracies)
    line_train.set_data(epochs, train_accuracies)

    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(0, 100)

    fig.canvas.draw()
    fig.canvas.flush_events()


start = 0

while epoch < EPOCHS-0.1:

    indices = np.random.permutation(len(x_train_aug))

    for start_index in range(0, len(x_train_aug), BATCH_SIZE):

        if time.perf_counter() - start > 2:
            show_accuracy()
            start = time.perf_counter()

        batch_indices = indices[start_index:start_index + BATCH_SIZE]

        batch_x = x_train_aug[batch_indices]
        batch_y = y_train_aug[batch_indices]

        AI.learn(batch_x, batch_y, LEARN_RATE, MOMENTUM)

        epoch += len(batch_x) / len(x_train_aug)

show_accuracy()

plt.close()