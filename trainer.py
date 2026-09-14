import time

import numpy as np
from matplotlib import pyplot as plt
import neural_network as nn
import utils

# AI CONFIGURATION
LEARN_RATE = 0.03
BATCH_SIZE = 100
MOMENTUM = 0.9
EPOCHS = 20

MAX_CHECK_SAMPLE = 1000 # for checking accuracy
RANDOMIZE_EXAMPLES = True
AUGMENT_EXAMPLES = True
SEARCH_EXAMPLES = 9

AI = nn.NeuralNetwork((utils.INPUT_SIZE, 128, 64, 10))
AI.randomize()

def init():

    utils.init()

    # 'one-hot encoding' encodes labels to a table of expected outputs. Use np.eye
    x_train = utils.x_train.reshape(utils.x_train.shape[0], -1)
    x_test = utils.x_test.reshape(utils.x_test.shape[0], -1)

    x_train_aug = utils.x_train_aug.reshape(utils.x_train_aug.shape[0], -1)
    y_train_aug = np.eye(10, dtype=np.float32)[utils.y_train_aug]

    ### SHOW EXAMPLES ###
    utils.show_examples(RANDOMIZE_EXAMPLES, AUGMENT_EXAMPLES, SEARCH_EXAMPLES)

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

        test_accuracy = AI.accuracy(x_test[:MAX_CHECK_SAMPLE], utils.y_test[:MAX_CHECK_SAMPLE]) * 100
        train_accuracy = AI.accuracy(x_train[:MAX_CHECK_SAMPLE], utils.y_train[:MAX_CHECK_SAMPLE]) * 100

        print(f"\nTest Accuracy: {test_accuracy : .2f}%")
        print(f"Training Accuracy: {train_accuracy : .2f}%")
        print(f"Epoch: {epoch:.2f}")

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

            if len(plt.get_figlabels()) == 0:
                return

            if time.perf_counter() - start > 2:
                show_accuracy()
                start = time.perf_counter()

            batch_indices = indices[start_index:start_index + BATCH_SIZE]

            batch_x = x_train_aug[batch_indices]
            batch_y = y_train_aug[batch_indices]

            AI.learn(batch_x, batch_y, LEARN_RATE, MOMENTUM)

            epoch += len(batch_x) / len(x_train_aug)

    show_accuracy()

    print("\nClass Accuracy:")
    test = AI.class_accuracy(x_test, utils.y_test)
    train = AI.class_accuracy(x_train, utils.y_train)
    for i, acc in enumerate(zip(test, train)):
        print(f"{i}). test: {acc[0]*100:.2f}  |  train: {acc[1]*100:.2f}%  |  diff: {(acc[0]-acc[1])*100:.2f}%")

    plt.close()