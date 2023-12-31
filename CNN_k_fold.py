# -*- coding: utf-8 -*-
"""CNN_naive_classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kJU-WoRlBpBIyTtZJK6tZ5bjR3uYEUKg
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from statistics import mean

acc_list = []
loss_list = []
i = 1
num_classes = 2

X = np.load('x_array_4d.npy')
y = np.load('y_array.npy')

# Split the data into training and validation sets
num_samples = X.shape[0]
indices = np.random.permutation(num_samples)
train_indices = indices[:int(0.9*num_samples)]
test_indices = indices[int(0.9*num_samples):]

x_train = X[train_indices]
y_train = y[train_indices]
x_test = X[test_indices]
y_test = y[test_indices]

# Convert the labels to one-hot encoding
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

print("x_train shape:")
print(x_train.shape)

print("y_train shape:")
print(y_train.shape)

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import KFold
num_folds = 9

kfold = KFold(n_splits=num_folds, shuffle=True)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



for fold, (train_indices, val_indices) in enumerate(kfold.split(x_train)):
    # Select the training and validation data for this fold
    x_train_fold, y_train_fold = x_train[train_indices], y_train[train_indices]
    x_val_fold, y_val_fold = x_train[val_indices], y_train[val_indices]

    # Define the CNN architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train_fold, y_train_fold,
                        validation_data=(x_val_fold, y_val_fold),
                        epochs=10, batch_size=64)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Fold {}: Test accuracy: {} , Test loss: {}'.format(fold+1, test_acc, test_loss))

    acc_list.append(test_acc)
    loss_list.append(test_loss)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model" + str(fold+1) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model" + str(fold+1) + ".h5")
    print("Saved model to disk")

print("Acc list:")
print(acc_list)
print("Average accuracy:", mean(acc_list))

print("Loss list:")
print(loss_list)
print("Loss accuracy:", mean(loss_list))
