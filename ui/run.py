import datetime
import math
import os
import sys
import time
import pandas as pd

import albumentations as AT
import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets
from PyQt5.QtChart import QChartView, QChart, QValueAxis, QSplineSeries
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from albumentations.pytorch import ToTensorV2

import gxipy as gx
from configs import MyConfig
from mainwindows import Ui_MainWindow
from models import get_model
from modules.xfeat import XFeat
from utils import get_colormap, transforms

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

coor_x, coor_y = -1, -1  # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
coor = np.array([[1, 1]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xfeat = XFeat().to(device)

config = MyConfig()
config.init_dependent_config()
model = get_model(config).to(device)
checkpoint = torch.load(config.load_ckpt_path, map_location=torch.device(device))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

colormap = torch.tensor(get_colormap(config)).to(device)
transform = AT.Compose([
    transforms.Scale(scale=0.5, is_testing=True),
    AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def OnMouseAction(event, x, y, flags, param):
    global coor_x, coor_y, coor
    if event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        print("%s" % x, y)
        coor_x, coor_y = x, y
        coor_m = [coor_x, coor_y]
        coor = np.row_stack((coor, coor_m))
    # elif event == cv2.EVENT_LBUTTONUP:
    #     cv2.line(old_frame, (coor_x, coor_y), (coor_x, coor_y), (255, 255, 0), 7)


def get_choose_action(img, OnMouseAction):
    while True:
        cv2.imshow('IImage', img)
        cv2.setMouseCallback('IImage', OnMouseAction)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):  # 空格完成退出操作
            break
    cv2.destroyAllWindows()  # 关闭页面


def output_choose_vedio(coor, frame):
    Video_choose = frame[coor[-2, 1] + 1:coor[-1, 1] - 1, coor[-2, 0] + 1:coor[-1, 0] - 1]
    # cv2.imshow('Video_choose', Video_choose)
    return Video_choose


class MainForm(Ui_MainWindow):
    def __init__(self, MainForm):
        super().__init__()
        self.prev_gray = None
        self.ref_precomp = None
        super().setupUi(MainForm)
        self.Height_choose = 0
        self.Width_choose = 0
        self.device_manager = None
        self.cam = None  # 相机
        self.cam_state = False  # 相机是否打开

        # 加载Qchart波形界面
        self.plot_qchart = QChartViewPlot()
        self.plot_qchart.setTitle("测量速度")
        self.plot_view.setChart(self.plot_qchart)
        self.plot_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.plot_view.setRubberBand(QChartView.RectangleRubberBand)

        # 按键状态初始化
        self.OpenCam.setEnabled(True)
        self.CloseCam.setEnabled(False)
        self.single.setEnabled(False)
        self.continuous.setEnabled(False)
        self.choose_roi.setEnabled(False)
        self.Trad_start.setEnabled(False)
        self.xFeat_start.setEnabled(False)

        # 槽信号连接
        self.OpenCam.clicked.connect(self.OpenCamera)
        self.single.clicked.connect(self.SingleAcq)
        self.CloseCam.clicked.connect(self.CloseCamera)
        self.continuous.clicked.connect(self.ContinuousAcq)
        self.choose_roi.clicked.connect(self.Choose_ROI)
        self.Trad_start.clicked.connect(lambda: self.measurement(method="Trad"))
        self.xFeat_start.clicked.connect(lambda: self.measurement(method="xFeat"))

        # 测速的一些参数
        self.track_len = 4  # 保存几帧特征点的坐标
        self.detect_interval = 1  # 过几帧检测一次角点
        self.tracks = []  # 存特征点的坐标
        self.frame_idx = 0
        self.d_sum = 0  # 每帧总距离
        self.d_ave = 0  # 每帧平均距离
        self.v = 0  # 每帧速度
        self.v_sum = 0  # 速度累加
        self.v_t = 0  # 平均速度
        self.num = 0  # 检测点数
        self.cost_time = 0  # 处理时间
        self.f = 0  # 帧数
        self.framerate_get = 0  # 采集帧率
        self.iters = 0  # 每多少帧输出一次速度
        self.angle = 95  # 筛选方向角度
        self.frame_counter = 0

        self.min_inliers = 50
        self.ransac_thr = 4.0
        self.H = None
        self.ref_precomp = None

        self.duration = 30  # 保存时间

        self.transform = 0.001942  # 每个像素代表的长度（m）

        # 图表绘制所需参数
        self.speeds = []  # 添加一个属性来存储每一帧的速度值
        self.frames = []  # 添加一个属性来存储对应的帧数
        self.operate_time = []  # 单帧处理时间
        self.fps = []  # 当前帧率
        self.detected = []  # 检测点数

    def OpenCamera(self):
        # 打开相机，获取相机基本信息
        self.cam_state = True

        # 初始化设备管理器
        self.device_manager = gx.DeviceManager()

        # 枚举设备，返回设备数量和设备信息列表
        dev_num, dev_info_list = self.device_manager.update_device_list()
        if dev_num == 0:
            self.show_dialog()

        else:
            # 获取第一个设备的序列号并打开设备
            # USB3.0相机没有ip

            # str_ip = dev_info_list[0].get("ip")
            # self.Info.append("第一台设备的ip:" + str(str_ip))

            self.Info.clear()
            str_sn = dev_info_list[0].get("sn")
            self.Info.append("第一台设备的SN:" + str(str_sn))
            str_id = dev_info_list[0].get("device_id")
            self.Info.append("第一台设备的id:" + str(str_id))
            # 打开设备
            self.cam = self.device_manager.open_device_by_sn(str_sn)

            self.framerate_get = self.cam.CurrentAcquisitionFrameRate.get()  # 获取当前采集的帧率
            self.iters = int(self.framerate_get)

            # 设置曝光模式为自动
            self.cam.ExposureMode.set(1)
            # 设置自动曝光最大限制
            self.cam.AutoExposureTimeMax.set(10000.0)
            # 设置自动增益
            self.cam.GainAuto.set(1)
            # 设置自动白平衡
            self.cam.BalanceWhiteAuto.set(1)
            # self.cam.BalanceRatio.set(100)    # 白平衡参数设置
            # self.cam.ExposuerTime.set(10.0)   # 曝光时间设置
            # self.cam.Gain.set(10.0)           # 增益参数设置

            # 按钮使能状态变化
            self.OpenCam.setEnabled(False)
            self.CloseCam.setEnabled(True)
            self.single.setEnabled(True)
            self.continuous.setEnabled(True)
            self.choose_roi.setEnabled(True)

    def show_dialog(self):
        QtWidgets.QMessageBox.critical(None, "Error", "No Devices Found")

    def CloseCamera(self):
        # 关闭相机
        self.OpenCam.setEnabled(True)
        self.CloseCam.setEnabled(False)
        self.single.setEnabled(False)
        self.continuous.setEnabled(False)
        self.choose_roi.setEnabled(False)
        self.Trad_start.setEnabled(False)
        self.xFeat_start.setEnabled(False)
        self.cam_state = False
        self.cam.stream_off()
        self.cam.close_device()

        self.plot_qchart.timer.stop()  # 关闭曲线显示

    def SingleAcq(self):
        self.cam.stream_on()
        self.Camera()
        self.cam.stream_off()

    def ContinuousAcq(self):
        self.single.setEnabled(False)
        self.continuous.setEnabled(False)
        self.choose_roi.setEnabled(False)
        self.cam.stream_on()

        while True:
            self.Camera()
            cv2.waitKey(1)

            if not self.cam_state:
                break

    def Camera(self):
        # 采集图像
        raw_image = self.cam.data_stream[0].get_image()
        # 将采集到的图像显示在self.Cam_show中
        Width = self.cam.Width.get()
        Height = self.cam.Height.get()
        # 从彩色原始图像获得RGB图像
        rgb_image = raw_image.convert("RGB")
        numpy_image = rgb_image.get_numpy_array()
        # 转为QImage格式
        pixmap = QImage(numpy_image, Width, Height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(pixmap)
        # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
        ratio = max(Width / self.Cam.width(), Height / self.Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.Cam.setPixmap(pixmap)

    def Choose_ROI(self):
        self.cam.stream_on()
        # 采集图像
        raw_image = self.cam.data_stream[0].get_image()
        Width = self.cam.Width.get()
        Height = self.cam.Height.get()
        # 从彩色原始图像获得RGB图像
        rgb_image = raw_image.convert("RGB")
        numpy_image = rgb_image.get_numpy_array()
        # 将图像从 RGB 转换为 BGR，因为 OpenCV 使用 BGR 格式
        bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # 选择框选区域
        get_choose_action(bgr_image, OnMouseAction)
        self.ref_frame = output_choose_vedio(coor, bgr_image)
        self.Width_choose = coor[-1, 0] - coor[-2, 0]  # 选中区域的宽
        self.Height_choose = coor[-1, 1] - coor[-2, 1]  # 选中区域的高
        print("视频选中区域的宽：%d" % self.Width_choose, '\n'"视频选中区域的高：%d" % self.Height_choose)

        cv2.rectangle(bgr_image, tuple(coor[-2, :] - 1), tuple(coor[-1, :]), (0, 255, 0), 2)  # 在原图像显示选定的框的范围
        # 将opencv格式转为QImage，在主窗口内显示原图像
        pixmap = self.CvMatToQImage(bgr_image)
        ratio = max(Width / self.Cam.width(), Height / self.Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.Cam.setPixmap(pixmap)

        # 停止采集流
        self.cam.stream_off()

        self.Trad_start.setEnabled(True)
        self.xFeat_start.setEnabled(True)

    def CvMatToQImage(self, cvMat):
        """
        将OpenCV图像转为QImage
        """
        if len(cvMat.shape) == 2:
            # 灰度图是单通道，所以需要用Format_Indexed8
            rows, columns = cvMat.shape
            bytesPerLine = columns
            return QImage(cvMat.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        else:
            rows, columns, channels = cvMat.shape
            bytesPerLine = channels * columns
            cvMat = cv2.cvtColor(cvMat, cv2.COLOR_BGR2RGB)
            pixmap = QImage(cvMat.data, columns, rows, bytesPerLine, QImage.Format_RGB888)
            return QPixmap.fromImage(pixmap)

    def measurement(self, method="Trad"):
        # 保存视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        last_start_time = time.time()
        out = cv2.VideoWriter("save_video/" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".mp4", fourcc,
                              24, (self.cam.Width.get(), self.cam.Height.get()))
        # 保存流速数据
        df = pd.DataFrame(columns=["时间", "流速v"])

        self.cam.stream_on()
        self.choose_roi.setEnabled(False)
        self.single.setEnabled(False)
        self.continuous.setEnabled(False)

        self.plot_qchart.timer.start()  # 开始曲线

        self.ref_precomp = xfeat.detectAndCompute(self.ref_frame, top_k=1024)[0]
        e1 = cv2.getTickCount()

        raw_image = self.cam.data_stream[0].get_image()
        rgb_image = raw_image.convert("RGB")
        numpy_image = rgb_image.get_numpy_array()
        bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        output = output_choose_vedio(coor, bgr_image)
        # 滤波+锐化
        frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
        frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
        frame_gray = cv2.filter2D(frame_gray, -1, kernel)
        self.prev_gray = frame_gray
        while True:
            self.del_files()
            # 每隔一段时间保存一段视频，判断是否需要分割视频
            if time.time() - last_start_time >= self.duration:
                out.release()
                out = cv2.VideoWriter("save_video/" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".mp4",
                                      fourcc, 24, (self.cam.Width.get(), self.cam.Height.get()))
                last_start_time = time.time()
            # 采集图像
            raw_image = self.cam.data_stream[0].get_image()
            self.frame_counter += 1
            date = datetime.datetime.now()
            timestamp = date.timestamp() * 100
            curr_time = timestamp
            Width = self.cam.Width.get()
            Height = self.cam.Height.get()
            # 从彩色原始图像获得RGB图像
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()

            # 实现分割
            pred_seg = self.seg(numpy_image)

            # 将图像从 RGB 转换为 BGR，因为 OpenCV 使用 BGR 格式
            bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            # 获得框选区域
            output = output_choose_vedio(coor, bgr_image)
            vis = output.copy()

            cv2.rectangle(bgr_image, tuple(coor[-2, :] - 1), tuple(coor[-1, :]), (0, 255, 0), 2)  # 在原视频显示选定的框的范围

            # 传统角点检测
            if method == "Trad":
                self.xFeat_start.setEnabled(False)
                # 滤波+锐化
                frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
                frame_gray = cv2.filter2D(frame_gray, -1, kernel)

                # curr_time = self.cam.get(cv2.CAP_PROP_POS_MSEC)  # 读取时间戳，用于计算单帧时间

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

                        temp = math.atan2(tr[-2][1] - tr[-1][1], tr[-2][0] - tr[-1][0]) / math.pi * 180  # 两帧之间特征点角度
                        dis = math.sqrt(math.pow(tr[-1][1] - tr[-2][1], 2) + math.pow(tr[-1][0] - tr[-2][0], 2))
                        if self.angle - 10 < temp < self.angle + 10 and dis > 3:  # 流动方向和速度筛选
                            new_tracks.append(tr)
                            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False,
                                  (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
                    if len(self.tracks) > 0:
                        self.d_ave = self.d_sum / len(self.tracks)
                    # 根据特征点数进行筛选，太多不行、太少也不行
                    if 1 < len(self.tracks) < 100:
                        self.num = self.num + 1
                        self.d_sum = 0
                        for pt in self.tracks:
                            dis = math.sqrt(math.pow(pt[-1][0] - pt[-2][0], 2) + math.pow(pt[-1][1] - pt[-2][1], 2))
                            self.d_sum = self.d_sum + dis
                        self.d_ave = self.d_sum / len(self.tracks)

                        self.v = self.d_ave / (curr_time - self.prev_time)
                        self.v_sum += self.v

                if self.frame_idx % self.detect_interval == 0:  # 每几帧检测一次特征点
                    mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
                    mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆
                        cv2.circle(mask, (x, y), 5, 0, -1)

                # 计算平均速度
                if self.frame_idx % self.iters == 0 and self.frame_idx != 0:
                    self.v_t = self.v_sum / (self.num + 0.00001)  # 避免除0
                    self.v_t = round(self.v_t, 6) * self.transform * 1000
                    self.v_sum = 0
                    self.num = 0

                # Shi-Tomasi角点检测
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中
                self.prev_gray = frame_gray
                self.prev_time = curr_time

            if method == "xFeat":
                self.Trad_start.setEnabled(False)
                self.iters = int(self.framerate_get / 2)
                new_pt1 = []
                new_pt2 = []
                output = cv2.GaussianBlur(output, (15, 15), 0)
                current = xfeat.detectAndCompute(output, top_k=1024)[0]

                kp1, des1 = current['keypoints'], current['descriptors']
                kp2, des2 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']

                idx0, idx1 = xfeat.match(des1, des2, 0.82)
                points1 = kp1[idx0].cpu().numpy()
                points2 = kp2[idx1].cpu().numpy()

                for (x1, y1), (x2, y2) in zip(points1, points2):
                    temp = math.atan2(y2 - y1, x2 - x1) / math.pi * 180
                    dis = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))

                    if self.angle - 10 < temp < self.angle + 10 and dis > 5:  # 流动方向和速度筛选
                        new_pt1.append((x1, y1, dis))
                        new_pt2.append((x2, y2))
                if len(new_pt1) > 10:
                    for (x1, y1, dis), (x2, y2) in zip(new_pt1, new_pt2):
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

                self.prev_time = curr_time
                self.ref_precomp = current

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(bgr_image, "plan1", (50, 50), font, 1, (0, 0, 255), 2)

            cv2.putText(bgr_image, "Speed v: " + str(round(self.v_t / 2.7, 1)) + "m/s", (50, 100), font, 1,
                        (0, 255, 0), 2)  # 比例记得删 ！！！！！！！！！！！！！！！！！！！！
            self.plot_qchart.y = round(self.v_t / 2.7, 1)
            # self.plot_qchart.x = curr_time
            cv2.putText(bgr_image, "Operating_time: " + str(round(self.cost_time, 3)) + "ms", (50, 150), font, 1,
                        (0, 255, 0), 2)

            # 图表数据
            if self.frame_idx % self.iters == 0 and self.frame_idx != 0:
                e2 = cv2.getTickCount()
                c_time = (e2 - e1) / cv2.getTickFrequency()
                self.cost_time = c_time * 1000 / self.iters
                e1 = e2
                self.speeds.append(self.v_t)
                self.frames.append(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
                new_rows = pd.Series(
                    {"时间": datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), "流速v": round(self.v_t / 2.7, 1)})
                df = pd.concat([df, new_rows.to_frame().T], ignore_index=True)
                df.to_csv("save/realtime_speed.csv", index=False)

            self.frame_idx += 1
            out.write(bgr_image)

            # 在大窗口中显示原视频
            pixmap = self.CvMatToQImage(bgr_image)
            ratio = max(Width / self.Cam.width(), Height / self.Cam.height())
            pixmap.setDevicePixelRatio(ratio)
            self.Cam.setPixmap(pixmap)

            # 在小窗口中显示ROI
            seg_pixmap = self.CvMatToQImage(pred_seg)
            roi_ratio = max(Width / self.Cam.width(), Height / self.Cam.height())
            seg_pixmap.setDevicePixelRatio(roi_ratio)
            self.Cam_Seg.setPixmap(seg_pixmap)
            cv2.waitKey(1)

            if not self.cam_state:
                break

    def seg(self, frame_rgb):
        """
        :param frame_rgb: Input RGB Image
        :return: Predict Segmentation
        """
        augmented = transform(image=frame_rgb)
        images_aug = augmented['image'].to(device, dtype=torch.float32).unsqueeze(0)

        preds = model(images_aug)
        preds = colormap[preds.max(dim=1)[1]].cpu().numpy()
        for i in range(preds.shape[0]):
            preds = preds[i].astype(np.uint8)
        preds = cv2.cvtColor(preds, cv2.COLOR_RGB2BGR)
        return preds

    def del_files(self):
        """
        判断文件数量是否超过设定值，如果超过，则删除一定数量的文件
        :return:
        """
        # 根据目录获取文件列表
        VIDEO_FILE_PATH = "save_video"
        files = os.listdir(VIDEO_FILE_PATH)
        MAX_FILES_COUNT = 2
        # 判断文件数量，如果超过了设定的最大值MAX_FILES_COUNT（自行定义），则删除最前面的几个文件
        if len(files) > MAX_FILES_COUNT:
            for i in files[:len(files) - MAX_FILES_COUNT]:
                os.remove(f'{VIDEO_FILE_PATH}\\{i}')


class QChartViewPlot(QChart):
    def __init__(self, parent=None):
        super(QChartViewPlot, self).__init__()
        self.series = QSplineSeries()
        self.series.setName("速度v")
        self.axisX = QValueAxis()
        self.axisY = QValueAxis()
        self.step = 500
        self.x = 0
        self.y = 0

        # 创建一个定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.handleTimeout)
        self.timer.setInterval(1000)

        redPen = QPen(Qt.red)
        redPen.setWidth(3)
        self.series.setPen(redPen)
        self.series.append(self.x, self.y)

        self.addSeries(self.series)

        self.addAxis(self.axisX, Qt.AlignBottom)
        self.addAxis(self.axisY, Qt.AlignLeft)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)
        self.axisX.setTickCount(10)
        self.axisX.setRange(-10, 0)
        self.axisY.setRange(-0.5, 10)

        # self.timer.start()

    def handleTimeout(self):
        x = self.plotArea().width() / self.axisX.tickCount()
        y = (self.axisX.max() - self.axisX.min()) / self.axisX.tickCount()
        self.x += y
        self.series.append(self.x, self.y)
        self.scroll(x, 0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()  # 主窗口
    w = MainForm(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
