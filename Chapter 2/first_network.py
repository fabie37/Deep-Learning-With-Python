""" So, this is the "hello world" example of
    deep learning systems. 
"""

# Loading mnist dataset into keras 
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# The network architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# The compliation step
network.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=['accuracy'])
