"""
运行后滚动鼠标滚轮放大缩小图像，按下esc确定，点两个点后按两次空格
！！！切记修改后面real_d乘的个数
"""


import cv2
import numpy as np
import math


def mouse_callback(event, x, y, flags, param):
    global scale
    global resized_image
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scale += 0.1  # 放大图像
        else:
            scale -= 0.1  # 缩小图像

        # 重新调整图像大小
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Image", resized_image)


image = cv2.imread("../../code/imgs/frame_0000.jpg")
resized_image = None
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)


def OnMouseAction(event, x, y, flags, param):  # 鼠标触发记录点位
    global coor_x, coor_y, coor
    if event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        print("%s" % x, y)
        coor_x, coor_y = x, y
        coor_m = [coor_x, coor_y]
        coor = np.row_stack((coor, coor_m))
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.line(resized_image, (coor_x, coor_y), (coor_x, coor_y), (255, 255, 0), 2)


def get_choose_action(img, OnMouseAction):
    while True:
        cv2.imshow('IImage', img)
        cv2.setMouseCallback('IImage', OnMouseAction)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):  # 空格完成退出操作
            break
    cv2.destroyAllWindows()  # 关闭页面


coor_x, coor_y, emptyImage = -1, -1, 0  # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
coor = np.array([[1, 1]])

scale = 1.0  # 初始缩放比例

while True:
    # cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mouse_callback)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下"Esc"键退出
        break

get_choose_action(resized_image, OnMouseAction)
cv2.line(resized_image, (coor[2, 0], coor[2, 1]), (coor[1, 0], coor[1, 1]), (255, 255, 0), 2)
cv2.imshow('image', resized_image)
cv2.waitKey(0)

real_d = 0.015 * 9  # 真实世界尺度
d = math.sqrt((coor[2, 1] - coor[1, 1])**2 + (coor[2, 0] - coor[1, 0])**2) / scale  # 像素尺度
ratio = real_d / d
print("像素尺寸:" + str(d))
print("实际尺寸:" + str(real_d))
print("缩放倍率:" + str(scale))
print("标定尺度:" + str(ratio))

cv2.destroyAllWindows()
