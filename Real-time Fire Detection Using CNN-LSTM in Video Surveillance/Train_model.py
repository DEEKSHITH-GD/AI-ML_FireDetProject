# set the matplotlib backend so figures can be saved in the background

# import the necessary packages
import os
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imutils import paths
import pickle
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
LABELS = set(["forest", "match stick", "stove"])
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
dataset='./dataset/'
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))
#print(imagePaths)
data = []
labels = []


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,validation_split=.10)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,validation_split=.10)


train_data = train_datagen.flow_from_directory(
    directory='./dataset/',
    target_size=(100, 100),
    class_mode='categorical',
    batch_size=32,
    subset="training", 
    shuffle=True,
    seed=42
)

valid_data = valid_datagen.flow_from_directory(
    directory='./dataset/',
    target_size=(100, 100),
    class_mode='categorical',
    batch_size=32,
    subset="validation", 
    shuffle=True,
    seed=42)



model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((1, 100, 100, 3), input_shape=(100, 100, 3)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
#model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50)
model.fit(train_data, validation_data=valid_data, epochs=10)
# serialize the model to disk
print("[INFO] serializing network...")
model.save("new_model.h5")
# serialize the label binarizer to disk
'''
pkl_filename = "new_model_leaves.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lb, file)

'''
