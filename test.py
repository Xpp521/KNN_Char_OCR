import cv2
from time import sleep

from main import find_roi
from setting import CAMERA_URI, RECORD, IMG_FOLDER, DATA_PATH, INIT_DATA_PATH, K


if __name__ == '__main__':
    img = cv2.imread('img0.png')
    cv2.imshow('O', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
    cv2.imshow('grey', gray)
    gray2 = cv2.dilate(gray, None, iterations=2)  # 膨胀2次
    gray2 = cv2.erode(gray2, None, iterations=2)  # 腐蚀2次
    cv2.imshow('B', gray2)
    edges = cv2.absdiff(gray, gray2)  # 做差
    cv2.imshow('diff', edges)

    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)  # x方向梯度图像
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)  # y方向梯度图像
    abs_x = cv2.convertScaleAbs(x)  # x方向梯度图像取绝对值
    # cv2.imshow('x', abs_x)
    abs_y = cv2.convertScaleAbs(y)  # y方向梯度图像取绝对值
    # cv2.imshow('y', abs_y)
    dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)  # 融合x、y两个梯度图像
    # cv2.imshow('dst', dst)
    ret, ddst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)  # 二值化
    cv2.imshow('ddst', ddst)


    cv2.waitKey(0)
