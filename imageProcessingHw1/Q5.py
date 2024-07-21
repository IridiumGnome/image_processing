from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import ssl
from tensorflow.keras.datasets import cifar10
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

ssl._create_default_https_context = ssl._create_unverified_context

class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Question 5")
        self.setGeometry(300, 250, 400, 300)

        layout = QVBoxLayout()
        self.setLayout(layout)

        groupBox = QGroupBox("VGG16 TEST")
        layout.addWidget(groupBox)

        groupBoxLayout = QVBoxLayout()
        groupBox.setLayout(groupBoxLayout)

        btn1 = QPushButton("Show Train Images")
        btn1.clicked.connect(self.Q1)
        #btn1.clicked.connect(self.Q1)
        btn2 = QPushButton("Show HyperParameter")
        btn2.clicked.connect(self.Q2)
        btn3 = QPushButton("Show Model Shortcut")
        btn3.clicked.connect(self.Q3)
        btn4 = QPushButton("Show Accuracy")
        btn4.clicked.connect(self.Q4)
        self.label = QLineEdit()
        btn5 = QPushButton("Test")
        btn5.clicked.connect(self.Q5)

        groupBoxLayout.addWidget(btn1)
        groupBoxLayout.addWidget(btn2)
        groupBoxLayout.addWidget(btn3)
        groupBoxLayout.addWidget(btn4)
        groupBoxLayout.addWidget(self.label)
        groupBoxLayout.addWidget(btn5)

    def Q1(self):
        class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        plt.figure(figsize  = (10, 10))
        for i in range(9):
            x = np.random.choice(range(len(train_labels)))
            plt.subplot(3, 3, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[x], cmap = plt.cm.binary)
            plt.xlabel(class_names[train_labels[x].item()])
            #plt.savefig('show_random.png')
        plt.show()

    def Q2(self):
        print('hyperparameters:')
        print('batch size: ', 100)
        print('learnig rate: ',0.1)
        print('optimizer: SGD')
    
    def Q3(self):
        with open('summary.txt', 'r') as f:
            print(f.read())
    
    def Q4(self):
        img = cv2.imread('result.png') 
        cv2.imshow('Result',img)
        
    def Q5(self):
        model = load_model('vgg16.h5')
    
        (train_image, train_label), (test_image, test_label) = cifar10.load_data()
    
        class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')
    
        if(self.label.text()): 
            num = int(self.label.text()) 
            img = test_image[num]
            
            #img = img_to_array(img)
            
            img = img / 255.0
            img = (np.expand_dims(img, 0)) 
            
            #
            #img = img.reshape(1, 32, 32, 3)
            #img = img.astype('float32')
            #
            
            prediction = model.predict(img)
            score = tf.nn.softmax(prediction[0])
            arr = prediction.flatten()
    
            print(
    "This image most likely belongs to {}.".format(class_names[np.argmax(score)]))
            #print(arr)
    
            plt.subplot(1,2,1)
            plt.imshow(test_image[num])
            plt.xlabel(class_names[test_label[num].item()])
            plt.subplot(1,2,2)
            plt.bar(class_names, arr)
            plt.show()
    
def application():
    app = QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    application()

