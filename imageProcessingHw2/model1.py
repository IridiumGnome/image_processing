# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from shutil import copyfile
import sys
import ssl
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import add, Dense, Conv2D, Flatten, Dropout, Conv1D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, ZeroPadding2D
from time import time

def my_resnet50(input_shape):

    input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)
    x = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer="he_normal", padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(64, (1, 1), strides=(1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (1, 1), strides=(2, 2), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (1, 1), strides=(2, 2), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (1, 1), strides=(2, 2), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2048, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2048, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=(2, 2), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(2048, (1, 1), kernel_initializer="he_normal", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=input, outputs=outputs, name='my_resnet50')
    return model


def Q1():
    INPUT_SHAPE = (224, 224, 3)
    model = my_resnet50(INPUT_SHAPE)
    model.summary()
    
def Q2():
    
    img = cv.imread('tensorboard.png')
    cv.imshow('Tensorboard', img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
    """
    INPUT_SHAPE = (224, 224, 3)
    model = my_resnet50(INPUT_SHAPE)
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=[
              keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    
    log_dir="logs/{}".format((time()))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
    history = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=validation_generator, validation_steps=STEP_SIZE_VALID, epochs=10, callbacks=[tensorboard_callback])
    model.save("ResNet2.h5")
    
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    print("Evaluate on test data")
    results = model.evaluate(test_generator, steps=STEP_SIZE_TEST)
    print("test loss, test acc:", results)
    """
    
    

def Q3(x):
    
    image_dict = {0: "cats", 1: "dogs"}
    if x > len(test_generator.filenames):
        print("Out of DataSet")
    else:
        ind1 = x // 64
        ind2 = x % 64
        plt.title(image_dict[np.argmax(pred[x])])
        print(image_dict[np.argmax(pred[x])])
        plt.imshow(test_generator[ind1][ind2])
        plt.show()
    
    
def Q4():
    data = {'augmentation method': 77, 'without augmentation': 74}
    Text = list(data.keys())
    Accuracy = list(data.values())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(Text, Accuracy, width=0.2)
    plt.title("Augmentation Comparison")
    plt.show()
    
    """
    train_datagen = ImageDataGenerator(rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2, 
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest',  
                                   rescale=1.0/255.)
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=64,
                                                    class_mode='binary',
                                                    target_size=(224, 224))

    validation_datagen = ImageDataGenerator(rotation_range = 40,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2, 
                                        zoom_range = 0.2,
                                        horizontal_flip = True,
                                        fill_mode = 'nearest',  
                                        rescale=1.0/255.)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=64,
                                                              class_mode='binary',
                                                              target_size=(224, 224))

    test_datagen = ImageDataGenerator(rescale=1.0/255.)
    test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                  batch_size=64,
                                                  class_mode='binary',
                                                  target_size=(224, 224))
    
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
    history = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=validation_generator, validation_steps=STEP_SIZE_VALID, epochs=10, callbacks=[tensorboard_callback])
    model.save("ResNet2.h5")
    
    print("Evaluate on test data")
    results = model.evaluate(test_generator, steps=STEP_SIZE_TEST)
    print("test loss, test acc:", results)
    
    """
    
new_model = load_model('ResNet2.h5')

TRAINING_DIR = "tmp/train/"
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=64,
                                                    class_mode='binary',
                                                    target_size=(224, 224),
                                                    shuffle=True)

VALIDATION_DIR = "tmp/validation/"
validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=64,
                                                              class_mode='binary',
                                                              target_size=(224, 224),
                                                              shuffle=True)

TEST_DIR = "tmp/test"
test_datagen = ImageDataGenerator(rescale=1.0/255.)
test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                  batch_size=64,
                                                  class_mode=None,
                                                  shuffle=False)

pred = new_model.predict(test_generator, verbose=1)



