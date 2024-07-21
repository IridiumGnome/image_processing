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
        self.setGeometry(300, 250, 600, 400)

        grid = QGridLayout()
        grid.addWidget(self.createQ1(), 0, 0)
        grid.addWidget(self.createQ2(), 0, 0)
        grid.addWidget(self.createQ3(), 0, 1)
        grid.addWidget(self.createQ4(), 1, 1)
        self.setLayout(grid)

    def createQ1(self):
        groupBox = QGroupBox("Find Contour")
        btn11 = QPushButton("Draw Contour")
        btn11.clicked.connect(self.Q11)
        btn12 = QPushButton("Count Rings")
        btn12.clicked.connect(self.Q12)
        self.label1 = QLabel("There are _ rings in img1.jpg")
        self.label2 = QLabel("There are _ rings in img2.jpg")

        vBox = QVBoxLayout()
        vBox.addWidget(btn11)
        vBox.addWidget(btn12)
        vBox.addWidget(self.label1)
        vBox.addWidget(self.label2)
        vBox.addStretch(1)
        groupBox.setLayout(vBox)

        return groupBox
    
    def Q11(self):
        img = cv2.imread("Q1_Image/img1.jpg")
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blurred, 30, 100)
        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

        img2 = cv2.imread("Q1_Image/img2.jpg")
        img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5) 
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        blurred2 = cv2.blur(img2,(11,11))
        blurred2 = cv2.medianBlur(blurred2,13)
        canny2 = cv2.Canny(blurred2, 30, 99)
        cnts2 = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
        cv2.drawContours(img2, cnts2, -1, (0, 255, 0), 2)
        
        cv2.imshow('Result', img)
        cv2.imshow('Result2', img2)

    def Q12(self):
        img = cv2.imread("Q1_Image/img1.jpg")
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blurred, 30, 100)
        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

        img2 = cv2.imread("Q1_Image/img2.jpg")
        img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5) 
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        blurred2 = cv2.blur(img2,(11,11))
        blurred2 = cv2.medianBlur(blurred2,13)
        canny2 = cv2.Canny(blurred2, 30, 99)
        cnts2 = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
        cv2.drawContours(img2, cnts2, -1, (0, 255, 0), 2)
        
        text1 = "There are {} rings in img1.jpg".format(len(cnts))
        text2 = "There are {} rings in img2.jpg".format(len(cnts2))
        self.label1.setText(text1)
        self.label2.setText(text2)
    
    def createQ2(self):
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

        return groupBox2
    
    def Q21(self):
        folder = "Q2_Image"
        for path in os.listdir(folder):
            img = cv2.imread(folder + '/' + path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
            img = cv2.resize(img, (800, 800))
            cv2.imshow("Result", img)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    def Q22(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        image_point = []
        objpoints = []
        objp = np.zeros((1, 11*8, 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        gray = None
        
        for N in range(1, 16):
            img = cv2.imread(f"Q2_Image/{str(N)}.bmp")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_point.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, image_point, gray.shape[::-1], None, None)
        print(mtx)

    def Q23(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
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
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
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
        print(dist)
        
    def Q25(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
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
        
        folder = "Q2_Image"
        for path in os.listdir(folder):
            img = cv2.imread(folder + '/' + path)
            h,  w = img.shape[:2]
            img2 = img.copy()
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
            img = cv2.resize(img, (400, 400))
            img2 = cv2.resize(img, (400, 400))
            numpy_horizontal_concat = np.concatenate((img, img2), axis=1)
            cv2.imshow('Result', numpy_horizontal_concat)
            #cv2.imshow("Result", img)
            cv2.waitKey(10000)
        cv2.destroyAllWindows()
    
    def createQ3(self):
        groupBox3 = QGroupBox("Augmented Reality")
        self.line3 = QLineEdit()
        btn31 = QPushButton("Show Words On Board")
        btn31.clicked.connect(self.Q31)
        btn32 = QPushButton("Show Words Vertically")
        btn32.clicked.connect(self.Q32)
        
        vBox3 = QVBoxLayout()
        vBox3.addWidget(self.line3)
        vBox3.addWidget(btn31)
        vBox3.addWidget(btn32)
        vBox3.addStretch(1)
        groupBox3.setLayout(vBox3)
 
        return groupBox3
    
    def Q31(self):
        folder = "Q3_Image"
        axis = np.float32([[3, 3, -3], [1, 1, 0], [3, 5, 0], [5, 1, 0]]).reshape(-1, 3)

        critical = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        corner_point_height = 8
        corner_point_width = 11

        possible_real_coordinate = np.zeros((corner_point_width * corner_point_height, 3), np.float32)
        width = 0
        height = 0
        for i in range(corner_point_height * corner_point_width):
            possible_real_coordinate[i, 0] = width
            width += 1
            possible_real_coordinate[i, 1] = height
            if width % corner_point_width == 0:
                width = 0
                height += 1
        # possible_real_coordinate = cwc(corner_point_width, corner_point_height)

        object_point = []
        image_point = []

        for path in os.listdir(folder):
            image = cv2.imread(folder + '/' + path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chess board corner
        ret, corner = cv2.findChessboardCorners(gray, (corner_point_width, corner_point_height), None,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH)

        if ret == True:
            # If corner was found
            object_point.append(possible_real_coordinate)

        image_point.append(corner)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_point, image_point, gray.shape[::-1], None, None)

        for path in os.listdir(folder):
            image = cv2.imread(folder + '/' + path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(folder + '/' + path + "\n")

            ret, corner = cv2.findChessboardCorners(gray, (corner_point_width, corner_point_height), None,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret == True:
                # If corner was found
                object_point.append(possible_real_coordinate)

            ret, rvecs, tvecs = cv2.solvePnP(possible_real_coordinate, corner, mtx, dist)

            tetrahedron_point, jacobian = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            for i in range(4):
                for j in range(4):
                    if j == i:
                        continue
                    img = cv2.line(image, tuple(tetrahedron_point[i].ravel()), tuple(tetrahedron_point[j].ravel()),
                                   (255, 0, 0), 5)
            img = cv2.resize(img, (1200, 1000))
            # cv2.imshow(str(name),image)

            cv2.imshow("Result", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    def Q32(self):
        pass
    
    def createQ4(self):
        groupBox4 = QGroupBox("Stereo Disparity Map")
        btn41 = QPushButton("Stereo Disparity Map")
        btn41.clicked.connect(self.Q41)

        vBox4 = QVBoxLayout()
        vBox4.addWidget(btn41)
        vBox4.addStretch(1)
        groupBox4.setLayout(vBox4)

        return groupBox4
    
    def Q41(self):
        imgL = cv2.imread('Q4_Image/imL.png',0)
        imgR = cv2.imread('Q4_Image/imR.png',0)

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR)
        
        normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        normalized = cv2.resize(normalized, (normalized.shape[1] // 4, normalized.shape[0] // 4))
        print(normalized.shape[1] // 4)
        print(normalized.shape[0] // 4)
        gray = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        cv2.imshow('gray', gray)
    
def application():
    app = QApplication(sys.argv)
    window = Window()
    
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
        application()
