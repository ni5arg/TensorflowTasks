# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:26:15 2020

@author: nisar
MNIST Fashion data classification Tensorflow
"""

import tensorflow as tf
#tf.enable_eager_execution()
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load("fashion_mnist", with_info=True,as_supervised=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

num_train_examples = metadata.splits["train"].num_examples
num_test_examples = metadata.splits["test"].num_examples
print("Number of training exampamples: {}".format(num_train_examples))
print("Number of test exampamples: {}".format(num_test_examples))

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))
# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#Now let's plot the first 25 examples in the test dataset
plt.figure(figsize = [10,10])
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(class_names[label])
    i += 1
plt.show()

#Build the model now
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape = (28,28,1)),
                             tf.keras.layers.Dense(128, activation = tf.nn.relu),
                             tf.keras.layers.Dense(10,activation = tf.nn.softmax)])

#compile the model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch= math.ceil(num_train_examples/BATCH_SIZE))

#Evaluate the model by predicting on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, steps=num_test_examples/BATCH_SIZE)
print("Accuracy on the test dataset = " ,test_accuracy)

#More specific prediction on random examples: Let's predict on one full batch
for test_images, test_labels in test_dataset.take(1): #Since we split test_dataset into batches earlier, this will
                                        #give us 1 batch, not just 1 example. Therefor we have 32 examples here.
    test_images = test_images.numpy()#Convert all 32 to numpy array because that's one of the accepted formats for model.predict
    test_labels = test_labels.numpy()
    prediction = model.predict(test_images)

prediction.shape#32 images, each row would have 10 columns, each column representing the probability of the image belonging to that class

#Final prediction for an image is the column with the largest number i.e. highest probability
#let's try on one image
np.argmax(prediction[13])#this gives us the class number, convert it to a string name
print("Predicted class = ", class_names[np.argmax(prediction[13])])
print("Actual class = ", class_names[test_labels[13]])

image =  test_images[...,0][13]
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.xlabel(class_names[test_labels[13]])
plt.show()