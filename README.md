## BACKGROUND:
  Yeah uhm, I made this only using basic libraries like numpy, cv2, matplotlib, and pygame.
  It isn't too amazing, but it's able to recognize most handritten digits trained from the MNIST dataset
  I now understand a lot more about gradient descent, backpropagation, vectorization, etc.
  I also learnt many functions available in numpy and matplotlib.

## NOTES:
  - It does some pre-processing to do some augmentation so it takes some time to load.
  - Close the examples window to continue to training.
  - The handwriting recognition program will run after all training is done

## FILES:
  - canvas.py - the interactive handwriting recognition program using pygame
  - comparison.py - comparing manual approximation of gradient descent to using backpropagation
  - neural_network.py - the neural network script and class
  - utils.py - utilities that reads the MNIST dataset and other stuff to help run the neural network
  - trainer.py - trains the AI and shows its progress using matplotlib
  - MNIST_ORG - the MNIST dataset

## FEATURES:
  - ReLU and Sigmoid activation functions (optional Tanh)
  - Momentum-based gradient descent
  - Data augmentation
  - Pygame-based interface
