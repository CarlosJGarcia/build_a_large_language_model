# perceptron-keras.ipynb
# 02/Apr/2026
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # suppress oneDNN warnings

import numpy as np
from keras import models
from keras import layers

print()
print("1. Carga de datos")
dataset = np.load("dataset.npz")
train_points  = dataset["train_points"]
train_labels  = dataset["train_labels"]
validation_points = dataset["validation_points"]
validation_labels = dataset["validation_labels"]
test_points = dataset["test_points"]
test_labels = dataset["test_labels"]

print()
print("2. Deficición del modelo y preparación de los datos")
# Sequential model, the simplest type of Keras model
network = models.Sequential()

# input_shape=[2]: Specifies the input is 2 values (1D array)
network.add(layers.Input(shape=(2,)))

# Core of the model: 1 Dense layer
# units=1: This specifies there is 1 neuron in this layer
# Activation function for the neuron: 'sigmoid'
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print()
print("3. Resumen del modelo")
network.summary()

# Training
print()
print("4. Model training")
network.fit(train_points, train_labels, epochs=10, batch_size=128, validation_data=(validation_points, validation_labels))

# Evaluación
print()
print("5. Evaluación con datos de test que el modelo no ha visto")
test_loss, test_acc = network.evaluate(test_points, test_labels)

print("======================================================")
print(f"Loss con datos de test: {test_loss}")
accu = test_acc * 100
print(f"Accuracy con datos de test: {test_acc} - {accu:.1f}%")

# Fin
print()
print("Fin del programa.")
