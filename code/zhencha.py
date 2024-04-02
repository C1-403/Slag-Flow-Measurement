'''
相邻帧间差分法直接对相邻的两帧图像做差分运算，并取差分运算的绝对值构成移动物体，优点是运算快速，实时性高，缺点是无法应对光照的突变，物体间一般具有空洞。
'''

# 导入必要的软件包
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
        cv2.line(old_frame, (coor_x, coor_y), (coor_x, coor_y), (255, 255, 0), 7)


def get_choose_action(img, OnMouseAction):
    while True:
        cv2.imshow('Roi', img)
        cv2.setMouseCallback('Roi', OnMouseAction)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):  # 空格完成退出操作
            break
    cv2.destroyAllWindows()  # 关闭页面


def output_choose_vedio(coor, frame):
    Video_choose = frame[coor[1, 1]:coor[2, 1], coor[1, 0]:coor[2, 0]]
    cv2.imshow('Video_choose', Video_choose)
    return Video_choose


# 视频文件输入初始化

coor_x, coor_y, emptyImage = -1, -1, 0  # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
coor = np.array([[1, 1]])

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码器
camera = cv2.VideoCapture('test2.mp4')  # 从文件读取视频,Todo:只需要修改成自己的视频路径即可进行测试
fps = camera.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

# 拿到第一帧图像
ret, old_frame = camera.read()
get_choose_action(old_frame, OnMouseAction)
Width_choose = coor[2, 0] - coor[1, 0]  # 选中区域的宽
Height_choose = coor[2, 1] - coor[1, 1]  # 选中区域的高
print("视频选中区域的宽：%d" % Width_choose, '\n'"视频选中区域的高：%d" % Height_choose)


def main():
    # 初始化当前帧的前帧
    lastFrame = None
    # 遍历视频的每一帧
    while camera.isOpened():
        # 读取下一帧
        (ret, frame) = camera.read()
        output = output_choose_vedio(coor, frame)
        cv2.rectangle(frame, tuple(coor[1, :]), tuple(coor[2, :]), (0, 255, 0), 2)  # 在原视频显示选定的框的范围
        cv2.imshow('lwpCVWindow', frame)  # 显示采集到的视频流

        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # 如果不能抓取到一帧，说明我们到了视频的结尾
        if not ret:
            break

            # 调整该帧的大小
        # frame = cv2.resize(frame, (500, 400), interpolation=cv2.INTER_CUBIC)

        # 如果第一帧是None，对其进行初始化
        if lastFrame is None:
            lastFrame = output
            continue

            # 计算当前帧和前帧的不同
        frameDelta = cv2.absdiff(lastFrame, output)

        # 当前帧设置为下一帧的前帧
        lastFrame = output.copy()

        # 结果转为灰度图
        # thresh = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)
        thresh = frameDelta


        # 图像二值化
        thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]

        ''' 
        #去除图像噪声,先腐蚀再膨胀(形态学开运算) 
        thresh=cv2.erode(thresh,None,iterations=1) 
        thresh = cv2.dilate(thresh, None, iterations=2) 
        '''

        # 阀值图像上的轮廓位置
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓
        for c in cnts:
            # 忽略小轮廓，排除误差
            if cv2.contourArea(c) < 30:
                continue

                # 计算轮廓的边界框，在当前帧中画出该框
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 显示当前帧
        cv2.imshow("frame", output)
        cv2.imshow("frameDelta", frameDelta)
        cv2.imshow("thresh", thresh)

        # 按esc退出
        if cv2.waitKey(250) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
