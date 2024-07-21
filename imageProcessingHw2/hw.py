# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import os.path
import sys


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("HW1")
        self.setGeometry(300, 250, 400, 200)

        groupBox2 = QGroupBox("Calibration")
        btn21 = QPushButton("Find Corners")
        btn21.clicked.connect(self.Q21)
        btn22 = QPushButton("Find Intrinsic")
        btn22.clicked.connect(self.Q22)
        
        self.line23 = QLineEdit()
        self.line23.setPlaceholderText("Select Image")
        btn23 = QPushButton("Find Extrinsic")
        btn23.clicked.connect(self.Q23)
        
        btn24 = QPushButton("Find Distortion")
        btn24.clicked.connect(self.Q24)
        btn25 = QPushButton("Show Result")
        btn25.clicked.connect(self.Q25)

        vBox2 = QVBoxLayout()
        vBox2.addWidget(btn21)
        vBox2.addWidget(btn22)
        vBox2.addWidget(self.line23)
        vBox2.addWidget(btn23)
        vBox2.addWidget(btn24)
        vBox2.addWidget(btn25)
        vBox2.addStretch(1)
        groupBox2.setLayout(vBox2)
        
        hbox = QHBoxLayout()
        hbox.addWidget(groupBox2)
        self.setLayout(hbox)
        
    def Q21(self):
        path = "Q2_Image"
        for name in os.listdir(path):
            img = cv2.imread(path + '/' + name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
            img = cv2.resize(img, (0,0), fx=0.3, fy=0.3) 
            cv2.imshow("Result", img)
            cv2.waitKey(500)
        cv2.destroyAllWindows()
        
    def Q22(self):
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        imgpoints = []
        objpoints = []
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        
        gray = None
        
        for N in range(1, 16):
            img = cv2.imread(f"Q2_Image/{str(N)}.bmp")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(mtx)
        
    def Q23(self):
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((1, 11*8, 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        gray = None
        
        for N in range(1, 16):
            img = cv2.imread(f"Q2_Image/{str(N)}.bmp")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        tmp = self.line23.text()
        num = (int(tmp))
        rmat, _ = cv2.Rodrigues(rvecs[num - 1])
        ext = np.concatenate((rmat, tvecs[num - 1]), 1)
        print(ext)
        
    def Q24(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        objpoints = [] 
        imgpoints = [] 
        
        gray = None
        
        for N in range(1, 16):
            img = cv2.imread(f"Q2_Image/{str(N)}.bmp")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(dist)
        
        
    def Q25(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        objpoints = [] 
        imgpoints = [] 
        
        gray = None
        
        for N in range(1, 16):
            img = cv2.imread(f"Q2_Image/{str(N)}.bmp")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
      
        path = "Q2_Image"
        for name in os.listdir(path):
            img = cv2.imread(path + '/' + name)
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            img = cv2.resize(img, (400, 400))
            img2 = cv2.resize(dst, (400, 400))
            numpy_horizontal_concat = np.concatenate((img, img2), axis=1)
            cv2.imshow('Result', numpy_horizontal_concat)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

def application():
    app = QApplication(sys.argv)
    window = Window()
    
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
        application()

