# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:38:46 2020

@author: nisar
mnist fashin dataset classification using CNN
"""

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#load the data
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

#Confirm the number of examples
num_train_example = metadata.splits['train'].num_examples
num_test_example = metadata.splits['test'].num_examples

#normalizing function to divide wach pixel value by 255
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    
    return image,label

#normalize the dataset
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#cache the datasets for faster execution
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

#Let's look at a random image
for image, label in train_dataset.take(1):
    break

image = image.numpy().reshape((28,28))
label = class_names[int(label)]

plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.xticks()
plt.yticks()
plt.xlabel(label)
plt.colorbar()
plt.grid(False)
plt.show()

#Now let's plot 25 samples from the test dataset
plt.figure(figsize = [10,10])
i = 0
for image,label in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.imshow(image, cmap = plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(class_names[label])
    i += 1

#build the model
model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3,3), padding='same' ,activation=tf.nn.relu, input_shape = (28,28,1)),
                            tf.keras.layers.MaxPool2D((2,2), strides=2),
                            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
                            tf.keras.layers.MaxPool2D((2,2), strides=2),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(128, activation=tf.nn.relu),
                            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#train the model
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_example).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs = 10, steps_per_epoch=math.ceil(num_train_example/BATCH_SIZE))

#Evaluate the model on test dataset now
test_loss, test_accuracy =  model.evaluate(test_dataset, steps=math.ceil(num_train_example/BATCH_SIZE))
print ("Accuracy on the test dataset = ", test_accuracy)

#Make prediction on random examples
for images, labels in test_dataset.take(1):
    images = images.numpy()
    labels = labels.numpy()
    predictions = model.predict(images)
    
predictions.shape#(32,10) because 1 take gives 1 batch of 32 images, each with 10 probabilities corresoponding to each class

np.argmax(predictions[12])#Check 12th image in this batch, which class has the max probability?
image = images[12].reshape((28,28))
predicted_label = class_names[np.argmax(predictions[12])]
actual_label= class_names[labels[12]]

plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.xlabel("Actual:" + actual_label + ", Predicted:" + predicted_label)







