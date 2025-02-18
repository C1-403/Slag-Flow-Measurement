# encoding:utf-8
import numpy as np
import cv2
import math
from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue
import torch


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
    return Video_choose


video_name = "24.avi"  # Todo：改成自己的视频名称
video_src = "../../../new_videos/" + video_name  # Todo:只需要修改成自己的视频路径即可进行测试
coor_x, coor_y, emptyImage = -1, -1, 0  # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
coor = np.array([[1, 1]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xfeat = XFeat().to(device)
# xfeat = LighterGlue().to(device)

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPG编码器
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
        # self.H = None
        self.e1 = None
        self.ref_frame = output_choose_vedio(coor, old_frame)
        self.ref_precomp = xfeat.detectAndCompute(self.ref_frame, top_k=1024)[0]
        self.prev_time = 0
        self.time = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.d_sum = 0  # 每帧总距离
        self.d_ave = 0  # 每帧平均距离
        self.v = 0  # 每帧速度
        self.v_t = 0  # 平均速度
        self.num = 0  # 检测点数
        self.cost_time = 0  # 处理时间
        # self.f = 0  # 帧数
        self.iters = fps  # 每多少帧输出一次速度
        self.angle = 95  # 筛选方向角度

        # self.min_inliers = 50
        # self.ransac_thr = 4.0

        # 图表绘制所需参数
        self.speeds = []  # 添加一个属性来存储每一帧的速度值
        self.frames = []  # 添加一个属性来存储对应的帧数
        self.operate_time = []  # 单帧处理时间
        self.fps = []  # 当前帧率
        self.detected = []  # 检测点数

        self.transform = 0.001942  # 每个像素代表的长度（m）

    def run(self):  # 光流运行方法
        self.e1 = cv2.getTickCount()
        while True:
            ret, frame = self.cam.read()  # 读取视频帧

            if ret:
                self.xfeat(frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:  # 按esc退出
                break

        self.time = [frames / fps for frames in self.frames]

    def xfeat(self, frame):
        output = output_choose_vedio(coor, frame)
        cv2.rectangle(frame, tuple(coor[1, :] - 1), tuple(coor[2, :]), (0, 255, 0), 1)  # 在原视频显示选定的框的范围

        curr_time = self.cam.get(cv2.CAP_PROP_POS_MSEC)  # 读取时间戳，用于计算单帧时间
        vis = output.copy()

        current = xfeat.detectAndCompute(output, top_k=1024)[0]

        kp1, des1 = current['keypoints'], current['descriptors']
        kp2, des2 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']

        idx0, idx1 = xfeat.match(des1, des2, 0.82)
        points1 = kp1[idx0].cpu().numpy()
        points2 = kp2[idx1].cpu().numpy()

        for (x1, y1), (x2, y2) in zip(points1, points2):
            temp = math.atan2(y2 - y1, x2 - x1) / math.pi * 180
            dis = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))

            if self.angle - 15 < temp < self.angle + 15:  # 流动方向和速度筛选
                cv2.circle(vis, (int(x1), int(y1)), 2, (0, 255, 0), -1)
                track = np.array([[x1, y1], [x2, y2]], np.int32)
                cv2.polylines(vis, [track], True, (0, 255, 0))
                self.d_sum = self.d_sum + dis
                self.num = self.num + 1

        if self.frame_idx % self.iters == 0 and self.frame_idx != 0:
            self.d_ave = self.d_sum / (self.num + 0.00001)
            self.v = self.d_ave / (curr_time - self.prev_time)
            self.v_t = round(self.v, 6) * self.transform * 1000

            self.d_sum = 0
            self.num = 0

        cv2.putText(frame, "plan1", (50, 50), self.font, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Speed v: " + str(round(self.v_t / 2.7, 1)) + "m/s", (50, 100), self.font, 1,
                    (0, 255, 0),
                    2)
        cv2.putText(frame, "Operating_time: " + str(round(self.cost_time, 3)) + "ms", (50, 150), self.font, 1,
                    (0, 255, 0), 2)
        cv2.imshow('lwpCVWindow', frame)  # 显示采集到的视频流
        cv2.imshow('lk_track', vis)

        # 图表数据
        if self.frame_idx % self.iters == 0 and self.frame_idx != 0:
            e2 = cv2.getTickCount()
            time = (e2 - self.e1) / cv2.getTickFrequency()
            self.cost_time = time * 1000 / self.iters
            # self.f = 1000 / self.cost_time
            self.e1 = e2

            self.operate_time.append(round(self.cost_time, 4))
            self.speeds.append(self.v_t)  # 记录速度
            # self.fps.append(self.f)
            self.frames.append(self.frame_idx)  # 记录帧数
            self.detected.append(len(points1))

        self.frame_idx += 1
        self.prev_time = curr_time
        self.ref_precomp = current


def main():
    App(video_src).run()
    camera.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
