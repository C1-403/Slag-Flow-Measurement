import cv2
import numpy as np


def OnMouseAction(event, x, y, flags, param):  # 鼠标触发记录点位
    global coor_x, coor_y, coor
    if event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        print("%s" % x, y)
        coor_x, coor_y = x, y
        coor_m = [coor_x, coor_y]
        coor = np.row_stack((coor, coor_m))
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.line(frame1, (coor_x, coor_y), (coor_x, coor_y), (255, 255, 0), 7)


def get_choose_action(img, OnMouseAction):
    while True:
        cv2.imshow('IImage', img)
        cv2.setMouseCallback('IImage', OnMouseAction)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):  # 空格完成退出操作
            break
    cv2.destroyAllWindows()  # 关闭页面


def output_choose_vedio(coor, frame):
    Video_choose = frame[coor[1, 1]:coor[2, 1], coor[1, 0]:coor[2, 0]]
    cv2.imshow('Video_choose', Video_choose)
    return Video_choose


coor_x, coor_y, emptyImage = -1, -1, 0  # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
coor = np.array([[1, 1]])

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码器
camera = cv2.VideoCapture('test2.mp4')  # 从文件读取视频,Todo:只需要修改成自己的视频路径即可进行测试
fps = camera.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
ret, frame1 = camera.read()  # 逐帧采集视频流
get_choose_action(frame1, OnMouseAction)

Width_choose = coor[2, 0] - coor[1, 0]  # 选中区域的宽
Height_choose = coor[2, 1] - coor[1, 1]  # 选中区域的高
print("视频选中区域的宽：%d" % Width_choose, '\n'"视频选中区域的高：%d" % Height_choose)


def main():
    output1 = output_choose_vedio(coor, frame1)
    prvs = cv2.cvtColor(output1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(output1)
    hsv[..., 1] = 255
    step = 20
    while 1:
        ret, frame2 = camera.read()
        output = output_choose_vedio(coor, frame2)
        next = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame2, tuple(coor[1, :]), tuple(coor[2, :]), (0, 255, 0), 2)  # 在原视频显示选定的框的范围
        cv2.imshow('lwpCVWindow', frame2)  # 显示采集到的视频流
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prvs = next

        # cv2.cartToPolar Calculates the magnitude and angle of 2D vectors.
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('optical_flow', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # 按esc退出
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
