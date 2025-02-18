# encoding:utf-8
"""''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt  # 添加matplotlib的导入

# lk光流金字塔参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 特征提取参数（qualityLevel表示特征点检测质量）
feature_params = dict(maxCorners=30,
                      qualityLevel=0.4,
                      minDistance=5,
                      blockSize=5)

# 锐化矩阵
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])


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


# out = cv2.VideoWriter("../output/videos/output.avi", fourcc, fps, (Width_choose, Height_choose))
# out2 = cv2.VideoWriter("../output/videos/origin.avi", fourcc, fps, (old_frame.shape[1], old_frame.shape[0]))


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
        self.frame_counter = 0

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

            self.frame_counter += 1
            if self.frame_counter == int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.frame_counter = 0
                self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if ret:
                self.measurement(frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:  # 按esc退出
                break

        self.time = [frames / fps for frames in self.frames]
        print(sum(self.speeds) / len(self.speeds))
        # self.draw_curve()

    def draw_curve(self):
        plt.figure(1)
        plt.plot(self.time, self.speeds, label='Speed over Time')
        plt.xlabel('Time(s)')
        plt.ylabel('Speed')
        plt.ylim(0, 8)
        plt.title('Speed Change Over Time')
        plt.legend()
        plt.savefig('../output/pictures/speed_curve1.jpg')

        plt.figure(2)
        plt.plot(self.time, self.operate_time, label='Operate time over Time')
        plt.xlabel('Time(s)')
        plt.ylabel('Operate time')
        plt.ylim(0, 15)
        plt.title('Operate time Over Time')
        plt.savefig('../output/pictures/operate_time_curve1.jpg')

        plt.figure(3)
        plt.plot(self.time, self.fps, label='FPS over Time')
        plt.xlabel('Time(s)')
        plt.ylabel('FPS')
        plt.ylim(0, 300)
        plt.title('FPS Change Over Time')
        plt.savefig('../output/pictures/fps_curve1.jpg')

        plt.figure(4)
        plt.plot(self.time, self.detected, label='Detected Numbers over Time')
        plt.xlabel('Time(s)')
        plt.ylabel('Detected Numbers')
        plt.ylim(0, 130)
        plt.title('Detected Numbers Change Over Time')
        plt.savefig('../output/pictures/detected_curve1.jpg')

    def measurement(self, frame):
        output = output_choose_vedio(coor, frame)
        e1 = cv2.getTickCount()

        cv2.rectangle(frame, tuple(coor[1, :] - 1), tuple(coor[2, :]), (0, 255, 0), 1)  # 在原视频显示选定的框的范围

        # 滤波+锐化
        frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
        # frame_gray = cv2.medianBlur(frame_gray, 5)
        # frame_gray = cv2.bilateralFilter(frame_gray, 15, 75, 75)
        frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
        frame_gray = cv2.filter2D(frame_gray, -1, kernel)

        curr_time = self.cam.get(cv2.CAP_PROP_POS_MSEC)  # 读取时间戳，用于计算单帧时间
        cv2.imshow("gray", frame_gray)
        vis = output.copy()

        if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                   **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                    **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系

            good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                if not good_flag:
                    continue
                tr.append((x, y))
                # self.d_sum = 0
                if len(tr) > self.track_len:
                    del tr[0]

                temp = math.atan2(tr[-1][1] - tr[-2][1], tr[-1][0] - tr[-2][0]) / math.pi * 180  # 两帧之间特征点角度
                dis = math.sqrt(math.pow(tr[-1][1] - tr[-2][1], 2) + math.pow(tr[-1][0] - tr[-2][0], 2))
                # print(temp)
                # print(tr)
                if self.angle - 20 < temp < self.angle + 20:  # 流动方向和速度筛选
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                # self.d_sum = self.d_sum + math.sqrt(math.pow(pre_x-x, 2) + math.pow(pre_y-y, 2))
            self.tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False,
                          (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
            if len(self.tracks) > 0:
                self.d_ave = self.d_sum / len(self.tracks)
            # print("d_ave:" + str(self.d_sum))
            # 根据特征点数进行筛选，太多不行、太少也不行
            if 1 < len(self.tracks) < 100:
                self.num = self.num + 1
                self.d_sum = 0
                for pt in self.tracks:
                    dis = math.sqrt(math.pow(pt[-1][0] - pt[-2][0], 2) + math.pow(pt[-1][1] - pt[-2][1], 2))
                    self.d_sum = self.d_sum + dis
                self.d_ave = self.d_sum / len(self.tracks)

                self.v = self.d_ave / (curr_time - self.prev_time)
                print(self.v)
                self.v_sum += self.v

            # print(self.tracks)
            # print(self.d_ave)
            # print(self.v)
            # print(self.frame_idx)

        if self.frame_idx % self.detect_interval == 0:  # 每几帧检测一次特征点
            mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
            mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆
                cv2.circle(mask, (x, y), 5, 0, -1)

        # 计算平均速度
        if self.frame_idx % self.iters == 0 and self.frame_idx != 0:
            self.v_t = self.v_sum / (self.num + 0.00001)  # 避免除0
            self.v_t = round(self.v_t, 6) * self.transform * 1000
            # print(self.v_t)
            self.v_sum = 0
            self.num = 0
            print("frames:" + str(self.frame_counter))

        # Shi-Tomasi角点检测
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

        # SIFT检测
        # sift = cv2.xfeatures2d.SIFT_create()
        # p = sift.detect(frame_gray, None)
        # if p is not None:
        #     for keypoint in p:
        #         x = keypoint.pt[0]
        #         y = keypoint.pt[1]
        #         self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

        # SURF关键点检测
        # surf = cv2.xfeatures2d.SURF_create()
        # p = surf.detect(frame_gray, None)
        # if p is not None:
        #     for keypoint in p:
        #         x = keypoint.pt[0]
        #         y = keypoint.pt[1]
        #         self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

        # ORB角点检测
        # orb = cv2.ORB_create()
        # p = orb.detect(frame_gray, None)
        # if p is not None:
        #     for keypoint in p:
        #         x = keypoint.pt[0]
        #         y = keypoint.pt[1]
        #         self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

        # FAST角点
        # fast = cv2.FastFeatureDetector_create()
        # p = fast.detect(frame_gray, None)
        # if p is not None:
        #     for keypoint in p:
        #         x = keypoint.pt[0]
        #         y = keypoint.pt[1]
        #         self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

        # 单帧图像处理时间
        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()

        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(vis, "Detected Numbers: " + str(len(self.tracks)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.putText(frame, "plan1", (50, 50), font, 1, (0, 0, 255), 2)
        # cv2.putText(frame, "Real time speed v: " + str(round(self.v, 1)), (50, 150), font, 1, (255, 255, 255), 2)

        # 图表数据
        if self.frame_idx % self.iters == 0 and self.frame_idx != 0:
            self.cost_time = time * 1000
            self.f = 1000 / self.cost_time

            self.operate_time.append(round(self.cost_time, 4))
            self.speeds.append(self.v_t)  # 记录速度
            self.fps.append(self.f)
            self.frames.append(self.frame_idx)  # 记录帧数
            self.detected.append(len(self.tracks))

        cv2.putText(frame, "Speed v: " + str(round(self.v_t / 2.7, 1)) + "m/s", (50, 100), font, 1, (0, 255, 0),
                    2)  # 比例记得删 ！！！！！！！！！！！！！！！！！！！！
        cv2.putText(frame, "Operating_time: " + str(round(self.cost_time, 3)) + "ms", (50, 150), font, 1,
                    (0, 255, 0), 2)
        # cv2.putText(frame, "FPS: " + str(int(self.f)), (50, 200), font, 1, (0, 255, 0), 2)
        # imgText = cv2.putText(imgText, "FPS:" + str(1 // time), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,
        # 0, 255))
        cv2.imshow('lwpCVWindow', frame)  # 显示采集到的视频流
        cv2.imshow('lk_track', vis)
        # if self.v > 17 or self.v < 6 or len(self.tracks) < 1:
        #     cv2.imwrite(f"../output/default_imgs/frame_ {self.frame_idx}.jpg", frame)
        # print("单帧处理时间:", round(time * 1000, 4), "ms")
        # print("FPS:", 1 // time)

        # print(self.detected)

        # out.write(vis)
        # out2.write(frame)

        self.frame_idx += 1
        self.prev_gray = frame_gray
        self.prev_time = curr_time


def main():
    App(video_src).run()
    camera.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
