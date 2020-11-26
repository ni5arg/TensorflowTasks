# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 14:52:31 2020

@author: nisar
Cats and dogs image classification. The images are RGB and can be any size.
In the process, we will build practical experience and develop intuition around the following concepts

Building data input pipelines using the tf.keras.preprocessing.image.ImageDataGenerator class — How can we efficiently work with data on disk to interface with our model?
Overfitting - what is it, how to identify it?
"""
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import os

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#we will make use of the class tf.keras.preprocessing.image.ImageDataGenerator 
#which will read data from disk. We therefore need to directly download Dogs vs. Cats 
#from a URL and unzip it

url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin = url, extract=True)

zip_dir_base = os.path.dirname(zip_dir)
#!find $zip_dir_base -type d -print #To see the directory tree

#Paths to all the different directories
base_dir = os.path.join(zip_dir_base, "cats_and_dogs_filtered")

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")

validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")

#Let's look at how many cats and dogs images we have in our training and validation directory
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('Total training cat images: ', num_cats_tr)
print('Total training dog images: ', num_dogs_tr)
print('Total validation cat images: ', num_cats_val)
print('Total validation dog images: ', num_dogs_val)
print('Total training images: ', total_train)
print('Total validation images: ', total_val)

#For convenience, we'll set up variables that will be used later 
#while pre-processing our dataset and training our network.
BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels


############    WITHOUT OVERFITTING MITIGATION #################################
#Pre-processing
#Read images from the disk
#Decode contents of these images and convert it into proper grid format as per their RGB content
#Convert them into floating point tensors
#Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=validation_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE),
                                                           class_mode='binary')

sample_training_images,labels = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1,5,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr,axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])

#Define the model
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150,3)),
        tf.keras.layers.MaxPool2D((2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        
        tf.keras.layers.Conv2D(128, (3,3), activation= 'relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2)
        ])
    
#Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#fit the model
EPOCHS=100
history = model.fit(train_data_gen, 
          epochs = EPOCHS, 
          validation_data=val_data_gen,
          validation_steps = int(np.ceil(total_val/float(BATCH_SIZE))),
          shuffle=True,
          steps_per_epoch=int(np.ceil(total_train/float(BATCH_SIZE))),
          )

#let's plot the accuracy and loss progression in each epoch
#First, prinnt which data fields are avaiable in the hostory object
print(history.history.keys())

acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = 'Training Accurancy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('./AccuracyLoss')
plt.show()
###############################################################################


#The above method shows 70% accuracy on validation set and 100% on the training set.
#The diverging graphs in accuracy and loss show overfitting.
#We'll try two methods to mitigate overfitting: Data augmentation and Dropout




