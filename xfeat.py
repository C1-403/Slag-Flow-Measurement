# encoding:utf-8
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt  # 添加matplotlib的导入


# 鼠标触发记录点位
def OnMouseAction(event, x, y, flags, param):
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
        cv2.imshow('IImage', img)
        cv2.setMouseCallback('IImage', OnMouseAction)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):  # 空格完成退出操作
            break
    cv2.destroyAllWindows()  # 关闭页面


def output_choose_vedio(coor, frame):
    Video_choose = frame[coor[1, 1]:coor[2, 1], coor[1, 0]:coor[2, 0]]
    # cv2.imshow('Video_choose', Video_choose)
    return Video_choose


video_name = "24.avi"  # Todo：改成自己的视频名称
video_src = "../../new_videos/" + video_name  # Todo:只需要修改成自己的视频路径即可进行测试
coor_x, coor_y, emptyImage = -1, -1, 0  # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
coor = np.array([[1, 1]])

fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPG编码器
camera = cv2.VideoCapture(video_src)  # 从文件读取视频
fps = camera.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

# 拿到第一帧图像
ret, old_frame = camera.read()
get_choose_action(old_frame, OnMouseAction)
Width_choose = coor[2, 0] - coor[1, 0]  # 选中区域的宽
Height_choose = coor[2, 1] - coor[1, 1]  # 选中区域的高
print("视频选中区域的宽：%d" % Width_choose, '\n'"视频选中区域的高：%d" % Height_choose)


class App:
    def __init__(self, video_src):  # 构造方法，初始化一些参数和视频路径
        self.time = None
        self.track_len = 4  # 保存几帧特征点的坐标
        self.detect_interval = 1  # 过几帧检测一次角点
        self.tracks = []  # 存特征点的坐标
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.d_sum = 0  # 每帧总距离
        self.d_ave = 0  # 每帧平均距离
        self.v = 0  # 每帧速度
        self.v_sum = 0  # 速度累加
        self.v_t = 0  # 平均速度
        self.num = 0  # 检测点数
        self.cost_time = 0  # 处理时间
        self.f = 0  # 帧数
        self.iters = fps  # 每多少帧输出一次速度
        self.angle = -95  # 筛选方向角度

        # 图表绘制所需参数
        self.speeds = []  # 添加一个属性来存储每一帧的速度值
        self.frames = []  # 添加一个属性来存储对应的帧数
        self.operate_time = []  # 单帧处理时间
        self.fps = []  # 当前帧率
        self.detected = []  # 检测点数
        self.transform = 0.001942  # 每个像素代表的长度（m）

    def run(self):  # 光流运行方法
        while True:
            ret, frame = self.cam.read()  # 读取视频帧

            if ret:
                output = output_choose_vedio(coor, frame)
                e1 = cv2.getTickCount()

                cv2.rectangle(frame, tuple(coor[1, :] - 1), tuple(coor[2, :]), (0, 255, 0), 1)  # 在原视频显示选定的框的范围

                # 滤波+锐化
                frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                curr_time = self.cam.get(cv2.CAP_PROP_POS_MSEC)  # 读取时间戳，用于计算单帧时间
                cv2.imshow("gray", frame_gray)
                vis = output.copy()

                e2 = cv2.getTickCount()
                time = (e2 - e1) / cv2.getTickFrequency()

                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame, "plan1", (50, 50), font, 1, (0, 0, 255), 2)

                cv2.putText(frame, "Speed v: " + str(round(self.v_t / 2.7, 1)) + "m/s", (50, 100), font, 1, (0, 255, 0),
                            2)

                # 图表数据
                if self.frame_idx % self.iters == 0 and self.frame_idx != 0:
                    self.cost_time = time * 1000
                    self.f = 1000 / self.cost_time

                    self.operate_time.append(round(self.cost_time, 4))
                    self.speeds.append(self.v_t)  # 记录速度
                    self.fps.append(self.f)
                    self.frames.append(self.frame_idx)  # 记录帧数
                    self.detected.append(len(self.tracks))

                cv2.putText(frame, "plan1", (50, 50), font, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Operating_time: " + str(round(self.cost_time, 3)) + "ms", (50, 150), font, 1,
                            (0, 255, 0), 2)
                cv2.imshow('lwpCVWindow', frame)  # 显示采集到的视频流
                cv2.imshow('lk_track', vis)

                self.frame_idx += 1
                self.prev_gray = frame_gray
                self.prev_time = curr_time

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:  # 按esc退出
                break

        self.time = [frames / fps for frames in self.frames]
        print(sum(self.speeds) / len(self.speeds))


def main():
    App(video_src).run()
    camera.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
