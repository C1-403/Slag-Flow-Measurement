# encoding:utf-8
'''''
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
'''

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt  # 添加matplotlib的导入

# from common import anorm2, draw_str
# from time import clock

lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=40,
                      qualityLevel=0.6,
                      minDistance=5,
                      blockSize=5)

kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])


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


video_name = "15.mp4"
video_src = "./test_videos/" + video_name  # Todo:只需要修改成自己的视频路径即可进行测试
coor_x, coor_y, emptyImage = -1, -1, 0  # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
coor = np.array([[1, 1]])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用XVID编码器
camera = cv2.VideoCapture(video_src)  # 从文件读取视频
fps = camera.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

# 拿到第一帧图像
ret, old_frame = camera.read()
get_choose_action(old_frame, OnMouseAction)
Width_choose = coor[2, 0] - coor[1, 0]  # 选中区域的宽
Height_choose = coor[2, 1] - coor[1, 1]  # 选中区域的高
print("视频选中区域的宽：%d" % Width_choose, '\n'"视频选中区域的高：%d" % Height_choose)


# out = cv2.VideoWriter("output/videos" + video_name, fourcc, 10, (Width_choose, Height_choose))


class App:
    def __init__(self, video_src):  # 构造方法，初始化一些参数和视频路径
        self.track_len = 2
        self.detect_interval = 4  # 过几帧检测一次角点
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.d_sum = 0
        self.d_ave = 0
        self.v = 0
        self.v_sum = 0
        self.iters = 110
        # 添加一个属性来存储每一帧的速度值
        self.speeds = []  
        # 添加一个属性来存储对应的帧数
        self.frames = []  
        # 单帧处理时间
        self.operate_time = []
        # 当前帧率
        self.fps = []

    def run(self):  # 光流运行方法
        while True:
            ret, frame = self.cam.read()  # 读取视频帧

            if ret:
                output = output_choose_vedio(coor, frame)
                e1 = cv2.getTickCount()

                cv2.rectangle(frame, tuple(coor[1, :]), tuple(coor[2, :]), (0, 255, 0), 1)  # 在原视频显示选定的框的范围
                cv2.imshow('lwpCVWindow', frame)  # 显示采集到的视频流
                frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                # frame_gray = cv2.medianBlur(frame_gray, 5)
                # frame_gray = cv2.bilateralFilter(frame_gray, 15, 75, 75)
                frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
                frame_gray = cv2.filter2D(frame_gray, -1, kernel)
                curr_time = self.cam.get(cv2.CAP_PROP_POS_MSEC)
                cv2.imshow("gray", frame_gray)
                vis = output.copy()

                if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪n
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
                        new_tracks.append(tr)
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                        # self.d_sum = self.d_sum + math.sqrt(math.pow(pre_x-x, 2) + math.pow(pre_y-y, 2))
                    self.tracks = new_tracks
                    # cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False,
                    #               (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
                    if len(self.tracks) > 0:
                        self.d_ave = self.d_sum / len(self.tracks)
                    # print("d_ave:" + str(self.d_sum))
                    if len(self.tracks) > 0:
                        self.d_sum = 0
                        for pt in self.tracks:
                            dis = math.sqrt(math.pow(pt[-1][0] - pt[-2][0], 2) + math.pow(pt[-1][1] - pt[-2][1], 2))
                            self.d_sum = self.d_sum + dis
                        self.d_ave = self.d_sum / len(self.tracks)
                    self.v = self.d_ave / (curr_time - self.prev_time)

                    # Draw velocity
                    self.speeds.append(self.v)  # 记录速度

                    self.frames.append(self.frame_idx)  # 记录帧数

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

                if self.frame_idx % self.iters == 0:
                    v_t = self.v_sum / self.iters
                    v_t = round(v_t, 6)
                    print(v_t)
                    self.v_sum = 0

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

                e2 = cv2.getTickCount()
                time = (e2 - e1) / cv2.getTickFrequency()
                imgText = cv2.putText(vis, "v:" + str(v_t), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                # imgText = cv2.putText(imgText, "FPS:" + str(1 // time), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,
                # 0, 255))
                cv2.imshow('lk_track', imgText)
                if self.frame_idx != 0:
                    print("单帧处理时间:", round(time*1000, 4), "ms")
                    self.operate_time.append(round(time*1000, 4))
                    print("FPS:", 1 // time)
                    self.fps.append(1 // time)
                    # out.write(imgText)

                self.frame_idx += 1
                self.prev_gray = frame_gray
                self.prev_time = curr_time
                

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:  # 按esc退出
                break

        self.draw_curve()

    def draw_curve(self):
        plt.figure(1)
        plt.plot(self.frames, self.speeds, label='Speed over frames')
        plt.xlabel('Frame number')
        plt.ylabel('Speed')
        plt.title('Speed Change Over Frames')
        plt.legend()
        plt.savefig('./output/pictures/speed_curve.jpg')

        plt.figure(2)
        plt.plot(self.frames, self.operate_time, label='Operate time over frames')
        plt.xlabel('Frame number')
        plt.ylabel('Operate time')
        plt.title('Operate time Over Frames')
        plt.savefig('./output/pictures/operate_time_curve.jpg')

        plt.figure(3)
        plt.plot(self.frames, self.fps, label='FPS over frames')
        plt.xlabel('Frame number')
        plt.ylabel('FPS')
        plt.title('FPS Change Over Frames')
        plt.savefig('./output/pictures/fps_curve.jpg')

        


def main():
    App(video_src).run()
    camera.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
