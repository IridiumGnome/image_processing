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
        self.setGeometry(300, 250, 500, 700)

        grid = QGridLayout()
        grid.addWidget(self.createQ1(), 0, 0)
        grid.addWidget(self.createQ2(), 1, 0)
        grid.addWidget(self.createQ3(), 0, 1)
        grid.addWidget(self.createQ4(), 1, 1)
        self.setLayout(grid)

    def createQ1(self):
        groupBox = QGroupBox("Image Processing")
        btn11 = QPushButton("Load Image")
        btn11.clicked.connect(self.Q11)
        btn12 = QPushButton("Color Separation")
        btn12.clicked.connect(self.Q12)
        btn13 = QPushButton("Color Transformation")
        btn13.clicked.connect(self.Q13)
        btn14 = QPushButton("Blending")
        btn14.clicked.connect(self.Q14)

        vBox = QVBoxLayout()
        vBox.addWidget(btn11)
        vBox.addWidget(btn12)
        vBox.addWidget(btn13)
        vBox.addWidget(btn14)
        vBox.addStretch(1)
        groupBox.setLayout(vBox)

        return groupBox

    def Q11(self):
        Sun_Img = cv2.imread("Q1_Image/Sun.jpg")
        cv2.imshow('image', Sun_Img)
        height = Sun_Img.shape[0]
        width = Sun_Img.shape[1]
        print("Height: ", height)
        print("Width: ", width)

    def Q12(self):
        img = cv2.imread("Q1_Image/Sun.jpg", 1)
        b, g, r = cv2.split(img)
        zeros = np.zeros(b.shape, np.uint8)
        b = cv2.merge([b,zeros,zeros])
        g = cv2.merge([zeros,g,zeros])
        r = cv2.merge([zeros,zeros,r])
        cv2.imshow("b", b)
        cv2.imshow("g", g)
        cv2.imshow("r", r)

    def Q13(self):
        img = cv2.imread("Q1_Image/Sun.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", gray)

    def Q14(self):

        def on_change(val):
            #print(val)
            alpha = val / 100
            beta = (1.0 - alpha)
            result = cv2.addWeighted(img2, alpha, img1, beta, 0.0)
            cv2.imshow("Blending", result)

        img1 = cv2.imread("Q1_Image/Dog_Strong.jpg")
        img2 = cv2.imread("Q1_Image/Dog_Weak.jpg")
        cv2.imshow("Blending", img2)
        cv2.createTrackbar("Slider", "Blending", 0, 100, on_change)
        blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    def createQ2(self):
        groupBox2 = QGroupBox("Image Smoothing")
        btn21 = QPushButton("Gaussian Blur")
        btn21.clicked.connect(self.Q21)
        btn22 = QPushButton("Bilateral filter")
        btn22.clicked.connect(self.Q22)
        btn23 = QPushButton("Median Filter")
        btn23.clicked.connect(self.Q23)

        vBox2 = QVBoxLayout()
        vBox2.addWidget(btn21)
        vBox2.addWidget(btn22)
        vBox2.addWidget(btn23)
        vBox2.addStretch(1)
        groupBox2.setLayout(vBox2)

        return groupBox2

    def Q21(self):
        img = cv2.imread("Q2_Image/Lenna_whiteNoise.jpg")
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow("Gaussian Blur", blur)

    def Q22(self):
        img = cv2.imread("Q2_Image/Lenna_whiteNoise.jpg")
        res = cv2.bilateralFilter(img, 9, 90, 90)
        cv2.imshow("Bilateral", res)

    def Q23(self):
        img = cv2.imread("Q2_Image/Lenna_pepperSalt.jpg")
        res1 = cv2.medianBlur(img, 3)
        cv2.imshow("Median 3x3", res1)
        res2 = cv2.medianBlur(img, 5)
        cv2.imshow("Median 5x5", res2)

    def createQ3(self):
        groupBox3 = QGroupBox("Edge Detection")
        btn31 = QPushButton("Gaussian Blur")
        btn31.clicked.connect(self.Q31)
        btn32 = QPushButton("Sobel X")
        btn32.clicked.connect(self.Q32)
        btn33 = QPushButton("Sobel Y")
        btn33.clicked.connect(self.Q33)
        btn34 = QPushButton("Magnitude")
        btn34.clicked.connect(self.Q34)
        
        vBox3 = QVBoxLayout()
        vBox3.addWidget(btn31)
        vBox3.addWidget(btn32)
        vBox3.addWidget(btn33)
        vBox3.addWidget(btn34)
        vBox3.addStretch(1)
        groupBox3.setLayout(vBox3)
 
        return groupBox3

    def Q31(self):
        img = cv2.imread("Q3_Image/House.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("Q3_Image/House_Gray.jpg", gray)
        cv2.imshow("Gray", gray)
        img2 = cv2.imread("Q3_Image/House_Gray.jpg")
        k = np.array([[1, 2, 1], 
                    [2, 4, 2], 
                    [1, 2, 1]])
        k = k/16
        blur = cv2.filter2D(img2, -1, k)
        cv2.imwrite("Q3_Image/Res.jpg", blur)
        cv2.imshow("New Gaussian", blur)

    def Q32(self):
        img = cv2.imread("Q3_Image/Res.jpg")
        SobelX = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
        res = cv2.filter2D(img, -1, SobelX)
        cv2.imwrite("Q3_Image/SobelX.jpg", res)
        cv2.imshow("SobelX", res)

    def Q33(self):
        img = cv2.imread("Q3_Image/Res.jpg")
        SobelY = np.array([[1, 2, 1], 
                        [0, 0, 0],
                        [-1, -2, -1]])
        res = cv2.filter2D(img, -1, SobelY)
        cv2.imwrite("Q3_Image/SobelY.jpg", res)
        cv2.imshow("SobelY", res)

    def Q34(self):
        img1 = cv2.imread("Q3_Image/SobelX.jpg",1)
        img2 = cv2.imread("Q3_Image/SobelY.jpg",1)
        res = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        #res = np.sqrt(img1**2 + img2**2)
        #cv2.imwrite("Q3_Image/34.jpg", res)
        #a = cv2.imread("Q3_Image/34.jpg")
        #norm = np.zeros(res.shape)
        norm = np.zeros((800,800))
        res = cv2.normalize(res, norm, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("Result", res)

    def createQ4(self):
        groupBox4 = QGroupBox("Transforms")
        btn41 = QLabel()
        btn41.setText("Resize")
        self.line41 = QLineEdit()
        self.line411 = QLineEdit()
        self.line41.setPlaceholderText("Width")
        self.line411.setPlaceholderText("Height")
        btn42 = QLabel()
        btn42.setText("Translation")
        self.line42 = QLineEdit()
        self.line422 = QLineEdit()
        self.line42.setPlaceholderText("X")
        self.line422.setPlaceholderText("Y")
        btn43 = QLabel()
        btn43.setText("Rotation & Scaling")
        self.line43 = QLineEdit()
        self.line43.setPlaceholderText("Angle")
        self.line433 = QLineEdit()
        self.line433.setPlaceholderText("Scale")
        self.line4333 = QLineEdit()
        self.line4333.setPlaceholderText("Window Size 400 300")
        btn44 = QLabel()
        btn44.setText("Shearing")
        self.line44 = QLineEdit()
        self.line44.setPlaceholderText("New: 10 100 200 50 100 250")
        self.line444 = QLineEdit()
        self.line444.setPlaceholderText("Old: 50 50 200 50 50 200")

        btn = QPushButton("Transform")
        btn.clicked.connect(self.func)

        vBox4 = QVBoxLayout()
        vBox4.addWidget(btn41)
        vBox4.addWidget(self.line41)
        vBox4.addWidget(self.line411)
        vBox4.addWidget(btn42)
        vBox4.addWidget(self.line42)
        vBox4.addWidget(self.line422)
        vBox4.addWidget(btn43)
        vBox4.addWidget(self.line43)
        vBox4.addWidget(self.line433)
        vBox4.addWidget(self.line4333)
        vBox4.addWidget(btn44)
        vBox4.addWidget(self.line44)
        vBox4.addWidget(self.line444)
        vBox4.addWidget(btn)
        vBox4.addStretch(1)
        groupBox4.setLayout(vBox4)

        return groupBox4

    def func(self):
        img = cv2.imread("Q4_Image/SQUARE-01.png")
        #cv2.namedWindow("Transformed", cv2.WINDOW_NORMAL)
        if(self.line41.text() and self.line411.text()):
            width = int(self.line41.text())
            height = int(self.line411.text())
            newSize = (width, height)
            res = cv2.resize(img, newSize, cv2.INTER_LINEAR)
            cv2.imwrite("Q4_Image/res.jpg", res)
        if(self.line42.text() and self.line422.text()):
            if(os.path.exists("Q4_Image/res.jpg")):
                img = cv2.imread("Q4_Image/res.jpg")
            x = int(self.line42.text())
            y = int(self.line422.text())
            height, width = img.shape[:2]
            T = np.array([[1, 0, x], [0, 1, y]], np.float32)
            res = cv2.warpAffine(img, T, (width+x, height+y))
            cv2.imwrite("Q4_Image/res.jpg", res)
        if(self.line43.text() and self.line433.text() and self.line4333.text()):
            if(os.path.exists("Q4_Image/res.jpg")):
                img = cv2.imread("Q4_Image/res.jpg")
            angle = int(self.line43.text())
            scale = float(self.line433.text())
            size = [int(x) for x in self.line4333.text().split() if x.isdigit()]
            height, width = img.shape[:2]
            center = (width/2, height/2)
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            res = cv2.warpAffine(img, matrix, (size[0], size[1]))
            cv2.imwrite("Q4_Image/res.jpg", res)
        if(self.line44.text() and self.line444.text()):
            if(os.path.exists("Q4_Image/res.jpg")):
                img = cv2.imread("Q4_Image/res.jpg")
            tmp1 = self.line44.text()
            tmp2 = [int(x) for x in tmp1.split() if x.isdigit()]
            newLocation = np.reshape(tmp2, (3, 2))
            tmp1 = self.line444.text()
            tmp2 = [int(x) for x in tmp1.split() if x.isdigit()]
            oldLocation = np.reshape(tmp2, (3, 2))
            pts1 = np.float32(oldLocation)
            pts2 = np.float32(newLocation)
            M = cv2.getAffineTransform(pts1, pts2)
            height, width = img.shape[:2]
            res = cv2.warpAffine(img, M, (width, height))
            #cv2.imwrite("Q4_Image/res.jpg", res)

        cv2.imshow("Transformed", res)
        if(os.path.exists("Q4_Image/res.jpg")):
            os.remove("Q4_Image/res.jpg")

def application():
    app = QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    application()

