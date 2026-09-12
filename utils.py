import numpy as np
import struct
from os.path import join
import random
import matplotlib.pyplot as plt
import cv2

input_path = 'MNIST_ORG'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

NOISE_STRENGTH = 0.3
NOISE_PROBABILITY = 0.3
SHIFT_STRENGTH = 7
SCALE_DOWN_STRENGTH = 0.5
SCALE_UP_STRENGTH = 0.2
ROTATION_STRENGTH = 30

IMG_WIDTH = 28
IMG_HEIGHT = 28

INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT
OUTPUT_SIZE = 10

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.frombuffer(file.read(), dtype=np.uint8)

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = np.frombuffer(file.read(), dtype=np.uint8)

        images = np.array(image_data, dtype=np.float32)/255
        images = images.reshape((size, rows, cols))

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        return (x_train, y_train), (x_test, y_test)


def augment_image(image):
    scale = np.random.uniform(1-SCALE_DOWN_STRENGTH, 1+SCALE_UP_STRENGTH)

    angle = np.random.uniform(-ROTATION_STRENGTH, ROTATION_STRENGTH)
    dx = np.random.randint(-SHIFT_STRENGTH, SHIFT_STRENGTH + 1)
    dy = np.random.randint(-SHIFT_STRENGTH, SHIFT_STRENGTH + 1)

    center = (IMG_WIDTH / 2, IMG_HEIGHT / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    M[0, 2] += dx
    M[1, 2] += dy

    image = cv2.warpAffine(image, M,
                           (IMG_WIDTH, IMG_HEIGHT),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)

    if NOISE_STRENGTH > 0 and NOISE_PROBABILITY > 0:
        noise = np.random.normal(0,NOISE_STRENGTH,image.shape).astype(np.float32)
        mask = np.random.random(noise.shape) < NOISE_PROBABILITY
        image = np.clip(image + noise * mask,0,1)

    return image


#
# Verify Reading Dataset via MnistDataloader class
#
#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(12,7))
    for index, x in enumerate(zip(images, title_texts)):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index+1)
        plt.imshow(image*255, cmap=plt.cm.gray)
        plt.title(title_text, fontsize = 8)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

#
# Load MNIST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train_aug = np.concatenate([x_train, x_train]) # 2x data

for i, x in enumerate(x_train_aug):
    x_train_aug[i] = augment_image(x)

y_train_aug = np.concatenate([y_train, y_train])

def get_random_train():
    r = random.randrange(len(x_train))
    return x_train[r]

def show_examples():
    images_2_show = []
    titles_2_show = []
    """
    ### RANDOM TRAINING IMAGES ###

    for i in range(15):
        r = random.randrange(len(x_train_aug))
        images_2_show.append(x_train_aug[r])
        titles_2_show.append(
            'training image [' + str(r) + '] = ' + str(y_train_aug[r])
        )
    """
    ### RANDDOM AUGMENTATIONS ###

    for i in range(15):
        r = 0
        images_2_show.append(augment_image(x_train[r]))
        titles_2_show.append(
            'training image [' + str(r) + '] = ' + str(y_train[r])
        )

    show_images(images_2_show, titles_2_show)