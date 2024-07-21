# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import ssl
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import model1 as f


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

        groupBox = QGroupBox("ResNet50")
        layout.addWidget(groupBox)

        groupBoxLayout = QVBoxLayout()
        groupBox.setLayout(groupBoxLayout)

        btn1 = QPushButton("Show Model Structure")
        btn1.clicked.connect(self.Q1)
        btn2 = QPushButton("Show TensorBoard")
        btn2.clicked.connect(self.Q2)
        btn3 = QPushButton("Test")
        btn3.clicked.connect(self.Q3)
        self.label = QLineEdit()
        btn4 = QPushButton("Data Augmentation")
        btn4.clicked.connect(self.Q4)

        groupBoxLayout.addWidget(btn1)
        groupBoxLayout.addWidget(btn2)
        groupBoxLayout.addWidget(btn3)
        groupBoxLayout.addWidget(self.label)
        groupBoxLayout.addWidget(btn4)

    def Q1(self):
        f.Q1()
    
    def Q2(self):
        img = cv2.imread('tensorboard.png') 
        cv2.imshow('Result',img)
    
    def Q3(self):
        txt = self.label.text()
        num = (int(txt))
        f.Q3(num)
    
    def Q4(self):
        f.Q4()
    
        
        
def application():
    app = QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    application()
