import datetime
import math
import os
import sys
import time
import pandas as pd
import glob
import albumentations as AT
import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtChart import QChartView, QChart, QValueAxis, QSplineSeries
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from albumentations.pytorch import ToTensorV2
import logging
import gxipy as gx
from configs import MyConfig
from mainwindows import Ui_mainWindow
from utils.splash.splash_ui import Ui_Form
from models import get_model
from modules.xfeat import XFeat
from utils import get_colormap, transforms
from numpy import ndarray
from van import *
from area_calculator import *

class QTextEditLogger(logging.Handler, QtCore.QObject):
    """
    Logging.Handler + QObject 多重继承。
    通过 signal_log 把日志消息发回主线程，再在槽里 append 到 QTextEdit。
    """
    signal_log = QtCore.pyqtSignal(str)

    def __init__(self, text_edit: QtWidgets.QTextEdit):
        logging.Handler.__init__(self)
        QtCore.QObject.__init__(self)
        self.text_edit = text_edit
        # 连接信号到槽
        self.signal_log.connect(self._append)

    def emit(self, record):
        msg = self.format(record)
        self.signal_log.emit(msg)

    @QtCore.pyqtSlot(str)
    def _append(self, msg):
        self.text_edit.append(msg)
        self.text_edit.ensureCursorVisible()


class SplashWithLog(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.textEdit.setReadOnly(True)

        handler = QTextEditLogger(self.ui.textEdit)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(fmt)

        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)

        logging.info("—— 应用开始启动 ——")
        logging.debug("准备进行初始化…")


def do_initialization():
    # lk光流金字塔参数
    logging.debug("【1】设置光流参数…")
    global lk_params, feature_params, kernel
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
    app.processEvents()

    global coor_x, coor_y, coor
    coor_x, coor_y = -1, -1  # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
    coor = np.array([[1, 1]])

    logging.debug("【2】初始化设备与特征提取模块…")
    global device, xfeat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xfeat = XFeat().to(device)
    app.processEvents()

    logging.debug("【3】加载配置…")
    global config
    config = MyConfig()
    config.init_dependent_config()
    app.processEvents()

    logging.debug("【4】构建并加载深度学习模型权重…")
    global model
    model = get_model(config).to(device)
    checkpoint = torch.load(config.load_ckpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    app.processEvents()

    logging.debug("【5】准备 colormap 与 数据变换…")
    global colormap, transform
    colormap = torch.tensor(get_colormap(config)).to(device)
    transform = AT.Compose([
        transforms.Scale(scale=0.5, is_testing=True),
        AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    app.processEvents()

    logging.info("—— 所有初始化完成 ——")


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


class FlowrateCaptureWorker(QObject):
    """运行 flowrate 相关方法的 worker"""
    start_capture = pyqtSignal()
    image_ready = pyqtSignal(ndarray)
    pixmap_ready = pyqtSignal(QPixmap)



    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def process_flowrate(self):
        if self.main.flowrate_cam_state:
            # 采集 + 转成 QPixmap ——
            self.main.flowrate_cam.stream_on()
            raw_image = self.main.flowrate_cam.data_stream[0].get_image()
            Width = self.main.flowrate_cam.Width.get()
            Height = self.main.flowrate_cam.Height.get()
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()
            pixmap = QImage(numpy_image, Width, Height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            self.pixmap_ready.emit(pixmap)
            self.main.flowrate_cam.stream_off()
            print("传了一次pixmap")

    @pyqtSlot()
    def bgrimage_capture(self):
        if self.main.flowrate_cam_state:
            print("采集图像")
            # 采集 + 转成 QPixmap ——
            self.main.flowrate_cam.stream_on()
            raw_image = self.main.flowrate_cam.data_stream[0].get_image()
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()
            bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            self.image_ready.emit(bgr_image)
            self.main.flowrate_cam.stream_off()

class FlowrateLoopWorker(QObject):
    """运行 flowrate 相关方法的 worker"""

    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def flow_caculate(self, method, out, df, e1, fourcc):
        last_start_time = time.time()
        # 把 image_ready 连接改成阻塞队列方式
        try:
            self.main.flowWorker.image_ready.disconnect()
        except TypeError:
            pass

        self.main.flowWorker.image_ready.connect(
            lambda bgr_image: self.main.serial_measurement(bgr_image, method, df, e1, out),
            Qt.BlockingQueuedConnection
        )
        while True:
            self.main.del_files()
            # 每隔一段时间保存一段视频，判断是否需要分割视频
            if time.time() - last_start_time >= self.main.duration:
                out.release()
                out = cv2.VideoWriter("save_video/" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".mp4",
                                      fourcc, 24, (self.main.flowrate_cam.Width.get(), self.main.flowrate_cam.Height.get()))
                last_start_time = time.time()

            try:
                self.main.flowWorker.start_capture.disconnect()
            except TypeError:
                pass
            self.main.flowWorker.start_capture.connect(self.main.flowWorker.bgrimage_capture)
            self.main.flowWorker.start_capture.emit()
            if not self.main.flowrate_cam_state:
                break


class FlowrateCaculateWorker(QObject):
    """运行 flowrate 相关方法的 worker"""

    start_caculate = pyqtSignal()
    done_caculate = pyqtSignal()
    done_prev_gray = pyqtSignal()

    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def serial_prev_gray(self, bgr_image: ndarray):
        output = output_choose_vedio(coor, bgr_image)
        # 滤波+锐化
        frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
        frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
        frame_gray = cv2.filter2D(frame_gray, -1, kernel)
        self.main.prev_gray = frame_gray
        self.done_prev_gray.emit()
        print("进入滤波计算")

    @pyqtSlot()
    def trad_caculate(self, output, vis, curr_time):
        print("进入传统计算")
        # 滤波+锐化
        frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
        frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
        frame_gray = cv2.filter2D(frame_gray, -1, kernel)
        # curr_time = self.cam.get(cv2.CAP_PROP_POS_MSEC)  # 读取时间戳，用于计算单帧时间

        if len(self.main.tracks) > 0:  # 检测到角点后进行光流跟踪
            img0, img1 = self.main.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.main.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                   **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                    **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系

            good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.main.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                if not good_flag:
                    continue
                tr.append((x, y))
                # self.d_sum = 0
                if len(tr) > self.main.track_len:
                    del tr[0]

                temp = math.atan2(tr[-2][1] - tr[-1][1], tr[-2][0] - tr[-1][0]) / math.pi * 180  # 两帧之间特征点角度
                dis = math.sqrt(math.pow(tr[-1][1] - tr[-2][1], 2) + math.pow(tr[-1][0] - tr[-2][0], 2))
                if self.main.angle - 10 < temp < self.main.angle + 10 and dis > 3:  # 流动方向和速度筛选
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            self.main.tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in self.main.tracks], False,
                          (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
            if len(self.main.tracks) > 0:
                self.main.d_ave = self.main.d_sum / len(self.main.tracks)
            # 根据特征点数进行筛选，太多不行、太少也不行
            if 1 < len(self.main.tracks) < 100:
                self.main.num = self.main.num + 1
                self.main.d_sum = 0
                for pt in self.main.tracks:
                    dis = math.sqrt(math.pow(pt[-1][0] - pt[-2][0], 2) + math.pow(pt[-1][1] - pt[-2][1], 2))
                    self.main.d_sum = self.main.d_sum + dis
                self.main.d_ave = self.main.d_sum / len(self.main.tracks)

                self.main.v = self.main.d_ave / (curr_time - self.main.prev_time)
                self.main.v_sum += self.main.v

        if self.main.frame_idx % self.main.detect_interval == 0:  # 每几帧检测一次特征点
            mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
            mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
            for x, y in [np.int32(tr[-1]) for tr in self.main.tracks]:  # 跟踪的角点画圆
                cv2.circle(mask, (x, y), 5, 0, -1)

        # 计算平均速度
        if self.main.frame_idx % self.main.iters == 0 and self.main.frame_idx != 0:
            self.main.v_t = self.main.v_sum / (self.main.num + 0.00001)  # 避免除0
            self.main.v_t = round(self.main.v_t, 6) * self.main.transform * 1000
            self.main.v_sum = 0
            self.main.num = 0

        # Shi-Tomasi角点检测
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.main.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中
        self.main.prev_gray = frame_gray
        self.main.prev_time = curr_time
        self.done_caculate.emit()

    @pyqtSlot()
    def xFeat_caculate(self, output, vis, curr_time):
        print("进入深度学习计算")
        self.main.iters = int(self.main.framerate_get / 2)
        new_pt1 = []
        new_pt2 = []
        output = cv2.GaussianBlur(output, (15, 15), 0)
        current = xfeat.detectAndCompute(output, top_k=1024)[0]

        kp1, des1 = current['keypoints'], current['descriptors']
        kp2, des2 = self.main.ref_precomp['keypoints'], self.main.ref_precomp['descriptors']

        idx0, idx1 = xfeat.match(des1, des2, 0.82)
        points1 = kp1[idx0].cpu().numpy()
        points2 = kp2[idx1].cpu().numpy()

        for (x1, y1), (x2, y2) in zip(points1, points2):
            temp = math.atan2(y2 - y1, x2 - x1) / math.pi * 180
            dis = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))

            if self.main.angle - 10 < temp < self.main.angle + 10 and dis > 5:  # 流动方向和速度筛选
                new_pt1.append((x1, y1, dis))
                new_pt2.append((x2, y2))
        if len(new_pt1) > 10:
            for (x1, y1, dis), (x2, y2) in zip(new_pt1, new_pt2):
                cv2.circle(vis, (int(x1), int(y1)), 2, (0, 255, 0), -1)
                track = np.array([[x1, y1], [x2, y2]], np.int32)
                cv2.polylines(vis, [track], True, (0, 255, 0))
                self.main.d_sum = self.main.d_sum + dis
                self.main.num = self.main.num + 1

        if self.main.frame_idx % self.main.iters == 0 and self.main.frame_idx != 0:
            self.main.d_ave = self.main.d_sum / (self.main.num + 0.00001)
            self.main.v = self.main.d_ave / (curr_time - self.main.prev_time)
            self.main.v_t = round(self.main.v, 6) * self.main.transform * 1000

            self.main.d_sum = 0
            self.main.num = 0

        self.main.prev_time = curr_time
        self.main.ref_precomp = current
        self.done_caculate.emit()

    def stop(self):
        # 停止线程
        self.main.flowrate_cam_state = False
        self.wait()

class CrossCaptureWorker(QObject):
    """运行 crosssect 相关方法的 worker"""
    start_capture = pyqtSignal()
    done_prev_gray = pyqtSignal()
    image_ready = pyqtSignal(ndarray)
    pixmap_ready = pyqtSignal(QPixmap)

    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def process_cross(self):
        if self.main.crosssect_cam_state:
            # 采集 + 转成 QPixmap ——
            self.main.crosssect_cam.stream_on()
            raw_image = self.main.crosssect_cam.data_stream[0].get_image()
            Width = self.main.crosssect_cam.Width.get()
            Height = self.main.crosssect_cam.Height.get()
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()
            pixmap = QImage(numpy_image, Width, Height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            self.pixmap_ready.emit(pixmap)
            self.main.crosssect_cam.stream_off()


    @pyqtSlot()
    def bgrimage_capture(self):
        if self.main.crosssect_cam_state:
            print("采集图像")
            # 采集 + 转成 QPixmap ——
            self.main.crosssect_cam.stream_on()
            raw_image = self.main.crosssect_cam.data_stream[0].get_image()
            arr = raw_image.get_numpy_array()
            bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.image_ready.emit(bgr_image)
            self.main.flowrate_cam.stream_off()



class CrossCaculateWorker(QObject):
    """运行 crosssect 相关方法的 worker"""

    done_caculate = pyqtSignal(float)


    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def single_crosssect_caculate(self, bgr_image: ndarray):
        area = van(bgr_image)
        self.done_caculate.emit(area)

class CrossLoopWorker(QObject):
    """运行 crosssect 相关方法的 worker"""

    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def crosssect_caculate(self):
        self.main.crosssect_cam_state = True
        self.main.crosssect_continuous.setEnabled(False)
        self.main.crosssect_CloseCam.setEnabled(True)
        self.main.crosssect_OpenCam.setEnabled(False)
        self.main.crosssect_start.setEnabled(True)
        self.main.bottom_face_start.setEnabled(True)
        self.main._start_ts = time.time()
        while True:
            try:
                self.main.crossWorker.start_capture.disconnect()
            except TypeError:
                pass
            try:
                self.main.crossWorker.image_ready.disconnect()
            except TypeError:
                pass
            try:
                self.main.CrossCaculateWorker.done_caculate.disconnect()
            except TypeError:
                pass
            self.main.CrossWorker.start_capture.connect(self.main.CrossWorker.bgrimage_capture)
            self.main.CrossWorker.image_ready.connect(
                lambda bgr_image: self.main.CrossCaculateWorker.single_crosssect_caculate(bgr_image))
            self.main.CrossWorker.image_ready.connect(lambda bgr_image: self.main.crosssect_view(bgr_image))
            self.main.CrossCaculateWorker.done_caculate.connect(lambda area: self.main.crosssect_plt(area))
            self.main.CrossWorker.start_capture.emit()
            if not self.main.crosssect_cam_state:
                break
            time.sleep(1)


class BottomfaceWorker(QObject):
    """运行 crosssect 相关方法的 worker"""
    start_capture = pyqtSignal()
    def __init__(self, main):
        super().__init__()
        self.main = main


    @pyqtSlot()
    def Bottomface_caculate(self):
        self.main.bottom_face_start.setEnabled(False)
        self.main.crosssect_cam_state = True

        if self.main.crosssect_cam_state:
            try:
                self.main.BottomfaceWorker.start_capture.disconnect(self.main.CrossWorker.bgrimage_capture)
            except TypeError:
                pass
            try:
                self.main.CrossWorker.image_ready.disconnect(lambda bgr_image: save_lower_countour(bgr_image,SAVE_ROI=False))
            except TypeError:
                pass
            self.main.BottomfaceWorker.start_capture.connect(self.main.CrossWorker.bgrimage_capture)
            self.main.CrossWorker.image_ready.connect(lambda bgr_image: save_lower_countour(bgr_image,SAVE_ROI=False))
            self.main.BottomfaceWorker.start_capture.emit()
        else:
            print("相机关闭，无法继续计算下表面面积。")

        time.sleep(1)
        self.main.bottom_face_start.setEnabled(True)


class MainForm(Ui_mainWindow):
    triggerFlow = pyqtSignal()

    def __init__(self, MainForm):
        super().__init__()
        self.prev_gray = None
        self.ref_precomp = None
        super().setupUi(MainForm)
        self.Height_choose = 0
        self.Width_choose = 0
        self.flowrate_device_manager = None
        self.flowrate_cam = None  # 相机
        self.flowrate_cam_state = False  # 相机是否打开
        self.crosssect_device_manager = None
        self.crosssect_cam = None
        self.crosssect_cam_state = False
        # 加载flowrate_Qchart波形界面
        self.flowrate_plot_qchart = flowrate_QChartViewPlot()
        self.flowrate_plot_qchart.setTitle("测量速度")
        self.flowrate_plot_qchart.setMargins(QMargins(20, 20, 20, 20))
        self.flowrate_plot_view.setChart(self.flowrate_plot_qchart)
        self.flowrate_plot_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.flowrate_plot_view.setRubberBand(QChartView.RectangleRubberBand)

        # 加载crosssect_Qchart波形界面
        self.crosssect_plot_qchart = crosssect_QChartViewPlot()
        self.crosssect_plot_qchart.setTitle("测量截面积")
        self.crosssect_plot_qchart.setMargins(QMargins(20, 20, 20, 20))
        self.crosssect_plot_view.setChart(self.crosssect_plot_qchart)
        self.crosssect_plot_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.crosssect_plot_view.setRubberBand(QChartView.RectangleRubberBand)

        # 按键状态初始化
        self.flowrate_OpenCam.setEnabled(True)
        self.flowrate_CloseCam.setEnabled(False)
        self.flowrate_single.setEnabled(False)
        self.flowrate_continuous.setEnabled(False)
        self.flowrate_choose_roi.setEnabled(False)
        self.flowrate_Trad_start.setEnabled(False)
        self.flowrate_xFeat_start.setEnabled(False)
        self.crosssect_OpenCam.setEnabled(True)
        self.crosssect_CloseCam.setEnabled(False)
        self.crosssect_continuous.setEnabled(False)
        self.crosssect_start.setEnabled(False)
        self.bottom_face_start.setEnabled(False)

        self.BottomfaceThread = QThread(self)
        self.BottomfaceWorker = BottomfaceWorker(self)
        self.BottomfaceWorker.moveToThread(self.BottomfaceThread)
        self.BottomfaceThread.start()

        self.CrossLoopThread = QThread(self)
        self.CrossLoopWorker = CrossLoopWorker(self)
        self.CrossLoopWorker.moveToThread(self.CrossLoopThread)
        self.CrossLoopThread.start()
        # 槽信号连接
        self.flowrate_OpenCam.clicked.connect(self.flowrate_OpenCamera)

        self.flowrate_single.clicked.connect(self.flowrate_SingleAcq)
        self.flowrate_CloseCam.clicked.connect(self.flowrate_CloseCamera)
        self.flowrate_continuous.clicked.connect(self.flowrate_ContinuousAcq)
        self.flowrate_choose_roi.clicked.connect(self.flowrate_Choose_ROI)
        self.flowrate_Trad_start.clicked.connect(lambda: self.measurement("Trad"))
        self.flowrate_xFeat_start.clicked.connect(lambda: self.measurement("xFeat"))

        self.crosssect_OpenCam.clicked.connect(self.crosssect_OpenCamera)
        self.crosssect_CloseCam.clicked.connect(self.crosssect_CloseCamera)
        self.crosssect_continuous.clicked.connect(self.crosssect_ContinuousAcq)
        self.crosssect_start.clicked.connect(self.CrossLoopWorker.crosssect_caculate)
        self.bottom_face_start.clicked.connect(self.BottomfaceWorker.Bottomface_caculate)

        # 计时器占位
        self._calc_timer = QTimer(self)
        self._calc_timer.timeout.connect(self._onCalcTimeout)
        self._start_ts = 0

        img_dir = os.path.join(os.getcwd(), "test")
        self._test_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self._test_index = 0
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

    def flowrate_OpenCamera(self):
        # 打开相机，获取相机基本信息
        self.flowrate_cam_state = True

        # 初始化设备管理器
        self.flowrate_device_manager = gx.DeviceManager()

        # 枚举设备
        dev_num, dev_info_list = self.flowrate_device_manager.update_device_list()
        if dev_num == 0:
            QMessageBox.critical(self, "Error", "未找到任何相机设备")
            return

        # 构造一个字符串列表，让用户选择
        items = []
        for idx, info in enumerate(dev_info_list):
            sn = info.get("sn")
            did = info.get("device_id")
            # 如果是 GigE 摄像头，也可以拿到 IP：info.get("ip")
            items.append(f"[{idx}] SN: {sn}   ID: {did}")

        # 弹出选择对话框
        item, ok = QInputDialog.getItem(
            self,
            "选择相机",
            "请选择要打开的相机：",
            items,
            0,  # 默认选中第一个
            False
        )
        if not ok:
            # 用户取消
            return

        # 解析选中的索引
        idx = int(item.split("]")[0].strip("["))
        dev = dev_info_list[idx]
        str_sn = dev.get("sn")
        str_id = dev.get("device_id")

        # 清空日志
        self.flowInfo.clear()
        self.flowInfo.append(f"已选择流速测量设备 → SN: {str_sn}")
        self.flowInfo.append(f"已选择流速测量设备设备 → ID: {str_id}")

        # 打开选中设备
        self.flowrate_cam = self.flowrate_device_manager.open_device_by_sn(str_sn)

        # 读取并输出当前帧率
        self.framerate_get = self.flowrate_cam.CurrentAcquisitionFrameRate.get()
        self.iters = int(self.framerate_get)

        # 自动曝光 / 自动增益 / 自动白平衡
        self.flowrate_cam.ExposureMode.set(1)
        self.flowrate_cam.AutoExposureTimeMax.set(10000.0)
        self.flowrate_cam.GainAuto.set(1)
        self.flowrate_cam.BalanceWhiteAuto.set(1)
        # self.flowrate_cam.BalanceRatio.set(100)    # 白平衡参数设置
        # self.flowrate_cam.ExposuerTime.set(10.0)   # 曝光时间设置
        # self.flowrate_cam.Gain.set(10.0)           # 增益参数设置

        # 切换按钮状态
        self.flowrate_OpenCam.setEnabled(False)
        self.flowrate_CloseCam.setEnabled(True)
        self.flowrate_single.setEnabled(True)
        self.flowrate_continuous.setEnabled(True)
        self.flowrate_choose_roi.setEnabled(True)

        self.flowThread = QThread(self)
        self.flowWorker = FlowrateCaptureWorker(self)
        self.flowWorker.moveToThread(self.flowThread)
        self.flowThread.start()

        self.flowCaculateThread = QThread(self)
        self.flowCaculateWorker = FlowrateCaculateWorker(self)
        self.flowCaculateWorker.moveToThread(self.flowCaculateThread)
        self.flowCaculateThread.start()

        self.flowLoopThread = QThread(self)
        self.flowLoopWorker = FlowrateLoopWorker(self)
        self.flowLoopWorker.moveToThread(self.flowLoopThread)
        self.flowLoopThread.start()

    def crosssect_OpenCamera(self):
        # 打开相机，获取相机基本信息
        self.crosssect_cam_state = True
        # 初始化设备管理器
        self.crosssect_device_manager = gx.DeviceManager()

        # 枚举设备
        dev_num, dev_info_list = self.crosssect_device_manager.update_device_list()
        if dev_num == 0:
            QMessageBox.critical(self, "Error", "未找到任何相机设备")
            return

        # 构造一个字符串列表，让用户选择
        items = []
        for idx, info in enumerate(dev_info_list):
            sn = info.get("sn")
            did = info.get("device_id")
            # 如果是 GigE 摄像头，也可以拿到 IP：info.get("ip")
            items.append(f"[{idx}] SN: {sn}   ID: {did}")

        # 弹出选择对话框
        item, ok = QInputDialog.getItem(
            self,
            "选择相机",
            "请选择要打开的相机：",
            items,
            0,  # 默认选中第一个
            False
        )
        if not ok:
            # 用户取消
            return

        # 解析选中的索引
        idx = int(item.split("]")[0].strip("["))
        dev = dev_info_list[idx]
        str_sn = dev.get("sn")
        str_id = dev.get("device_id")

        # 清空日志
        self.crossInfo.clear()
        self.crossInfo.append(f"已选择截面积计算设备 → SN: {str_sn}")
        self.crossInfo.append(f"已选择截面积计算设备 → ID: {str_id}")

        # 打开选中设备
        self.crosssect_cam = self.crosssect_device_manager.open_device_by_sn(str_sn)

        # 读取并输出当前帧率
        self.framerate_get = self.crosssect_cam.CurrentAcquisitionFrameRate.get()
        self.iters = int(self.framerate_get)

        # 自动曝光 / 自动增益 / 自动白平衡
        self.crosssect_cam.ExposureMode.set(1)
        self.crosssect_cam.AutoExposureTimeMax.set(10000.0)
        self.crosssect_cam.GainAuto.set(1)
        self.crosssect_cam.BalanceWhiteAuto.set(1)
        # self.crosssect_cam.BalanceRatio.set(100)    # 白平衡参数设置
        # self.crosssect_cam.ExposuerTime.set(10.0)   # 曝光时间设置
        # self.crosssect_cam.Gain.set(10.0)           # 增益参数设置

        # 切换按钮状态
        self.crosssect_OpenCam.setEnabled(False)
        self.crosssect_CloseCam.setEnabled(True)
        self.crosssect_continuous.setEnabled(True)
        self.crosssect_start.setEnabled(True)
        self.bottom_face_start.setEnabled(True)
        self.CrossThread = QThread(self)
        self.CrossWorker = CrossCaptureWorker(self)
        self.CrossWorker.moveToThread(self.CrossThread)
        self.CrossThread.start()

        self.CrossCaculateThread = QThread(self)
        self.CrossCaculateWorker = CrossCaculateWorker(self)
        self.CrossCaculateWorker.moveToThread(self.CrossCaculateThread)
        self.CrossCaculateThread.start()





    def flowrate_CloseCamera(self):
        # 关闭相机
        self.flowrate_OpenCam.setEnabled(True)
        self.flowrate_CloseCam.setEnabled(False)
        self.flowrate_single.setEnabled(False)
        self.flowrate_continuous.setEnabled(False)
        self.flowrate_choose_roi.setEnabled(False)
        self.flowrate_Trad_start.setEnabled(False)
        self.flowrate_xFeat_start.setEnabled(False)
        self.flowrate_cam_state = False
        self.flowrate_cam.stream_off()
        self.flowrate_cam.close_device()

        self.flowrate_plot_qchart.timer.stop()  # 关闭曲线显示

        self.flowThread.quit()
        self.flowThread.wait()
        self.flowCaculateThread.quit()
        self.flowCaculateThread.wait()

    def crosssect_CloseCamera(self):
        # 关闭相机
        self.crosssect_OpenCam.setEnabled(True)
        self.crosssect_CloseCam.setEnabled(False)
        self.crosssect_continuous.setEnabled(False)
        self.crosssect_start.setEnabled(False)
        self.bottom_face_start.setEnabled(False)
        self.crosssect_cam_state = False
        self.crosssect_cam.stream_off()
        self.crosssect_cam.close_device()
        self.crosssect_plot_qchart.timer.stop()  # 关闭曲线显示
        # # 清空图表数据
        # for series in self.crosssect_plot_qchart.chart.series():
        #     series.clear()
        # # 重置坐标轴范围（可选）
        # self.crosssect_plot_qchart.axisX.setRange(0, 10)
        # self.crosssect_plot_qchart.axisY.setRange(0, 200)
        self.CrossThread.quit()
        self.CrossThread.wait()
        self.CrossCaculateThread.quit()
        self.CrossCaculateThread.wait()

    def flowrate_SingleAcq(self):
        try:
            self.flowWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.flowWorker.pixmap_ready.disconnect()
        except TypeError:
            pass
        self.flowWorker.start_capture.connect(self.flowWorker.process_flowrate)
        self.flowWorker.pixmap_ready.connect(lambda pixmap: self.flowrate_Camera(pixmap))
        print(time.time())
        self.flowWorker.start_capture.emit()

    def flowrate_ContinuousAcq(self):
        try:
            self.flowWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.flowWorker.pixmap_ready.disconnect()
        except TypeError:
            pass
        self.flowWorker.start_capture.connect(self.flowWorker.process_flowrate)
        self.flowWorker.pixmap_ready.connect(lambda pixmap: self.flowrate_Camera(pixmap))
        self.flowrate_single.setEnabled(False)
        self.flowrate_continuous.setEnabled(False)
        self.flowrate_choose_roi.setEnabled(False)
        self.flowrate_cam_state = True
        while True:
            print(time.time())
            self.flowWorker.start_capture.emit()
            cv2.waitKey(1)

            if not self.flowrate_cam_state:
                break

    def crosssect_ContinuousAcq(self):
        try:
            self.CrossWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.CrossWorker.pixmap_ready.disconnect()
        except TypeError:
            pass
        self.CrossWorker.start_capture.connect(self.CrossWorker.process_cross)
        self.CrossWorker.pixmap_ready.connect(lambda pixmap: self.cross_Camera(pixmap))
        self.crosssect_continuous.setEnabled(False)
        self.crosssect_cam_state = True
        while True:
            self.CrossWorker.start_capture.emit()
            cv2.waitKey(1)

            if not self.crosssect_cam_state:
                break

    def flowrate_Camera(self, pixmap: QPixmap):
        """
        接收到子线程传来的 QPixmap，用于更新 UI
        """
        # 获取图像的宽高比，使其适应显示窗口
        Width = self.flowrate_cam.Width.get()
        Height = self.flowrate_cam.Height.get()
        ratio = max(Width / self.flowrate_Cam.width(), Height / self.flowrate_Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.flowrate_Cam.setPixmap(pixmap)

    def cross_Camera(self, pixmap: QPixmap):
        """
        接收到子线程传来的 QPixmap，用于更新 UI
        """
        # 获取图像的宽高比，使其适应显示窗口
        Width = self.crosssect_cam.Width.get()
        Height = self.crosssect_cam.Height.get()
        ratio = max(Width / self.crosssect_Cam.width(), Height / self.crosssect_Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.crosssect_Cam.setPixmap(pixmap)


    def flowrate_Choose_ROI(self):

        self.flowWorker.start_capture.connect(self.flowWorker.bgrimage_capture)
        self.flowWorker.image_ready.connect(lambda bgr_image:self.serial_flowrate_Choose_ROI(bgr_image))
        self.flowWorker.start_capture.emit()

    def serial_flowrate_Choose_ROI(self, bgr_image: ndarray):
        Width = self.flowrate_cam.Width.get()
        Height = self.flowrate_cam.Height.get()
        get_choose_action(bgr_image, OnMouseAction)
        self.ref_frame = output_choose_vedio(coor, bgr_image)
        self.Width_choose = coor[-1, 0] - coor[-2, 0]  # 选中区域的宽
        self.Height_choose = coor[-1, 1] - coor[-2, 1]  # 选中区域的高
        print("视频选中区域的宽：%d" % self.Width_choose, '\n'"视频选中区域的高：%d" % self.Height_choose)

        cv2.rectangle(bgr_image, tuple(coor[-2, :] - 1), tuple(coor[-1, :]), (0, 255, 0), 2)  # 在原图像显示选定的框的范围
        # 将opencv格式转为QImage，在主窗口内显示原图像
        pixmap = self.CvMatToQImage(bgr_image)
        ratio = max(Width / self.flowrate_Cam.width(), Height / self.flowrate_Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.flowrate_Cam.setPixmap(pixmap)

        # 停止采集流
        self.flowrate_cam.stream_off()

        self.flowrate_Trad_start.setEnabled(True)
        self.flowrate_xFeat_start.setEnabled(True)

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



    def serial_measurement(self, bgr_image: ndarray, method, df, e1, out):
        print("进入计算")
        self.frame_counter += 1
        date = datetime.datetime.now()
        timestamp = date.timestamp() * 100
        curr_time = timestamp

        # 获得框选区域
        output = output_choose_vedio(coor, bgr_image)
        vis = output.copy()
        cv2.rectangle(bgr_image, tuple(coor[-2, :] - 1), tuple(coor[-1, :]), (0, 255, 0), 2)  # 在原视频显示选定的框的范围
        # 传统角点检测
        if method == "Trad":
            self.flowrate_xFeat_start.setEnabled(False)

            # curr_time = self.cam.get(cv2.CAP_PROP_POS_MSEC)  # 读取时间戳，用于计算单帧时间
            try:
                self.flowCaculateWorker.start_caculate.disconnect()
            except TypeError:
                pass
            try:
                self.flowCaculateWorker.done_caculate.disconnect()
            except TypeError:
                pass
            self.flowCaculateWorker.start_caculate.connect(lambda: self.flowCaculateWorker.trad_caculate(output, vis, curr_time))
            self.flowCaculateWorker.done_caculate.connect(lambda: self.caculate_view(bgr_image, df, e1, out))
            self.flowCaculateWorker.start_caculate.emit()
            print("发送计算信号")

        if method == "xFeat":
            self.flowrate_Trad_start.setEnabled(False)
            try:
                self.flowCaculateWorker.start_caculate.disconnect()
            except TypeError:
                pass
            try:
                self.flowCaculateWorker.done_caculate.disconnect()
            except TypeError:
                pass
            self.flowCaculateWorker.start_caculate.connect(lambda: self.flowCaculateWorker.xFeat_caculate(output, vis, curr_time))
            self.flowCaculateWorker.done_caculate.connect(lambda: self.caculate_view(bgr_image, df, e1, out))
            self.flowCaculateWorker.start_caculate.emit()
        print(time.time())

    def caculate_view(self, bgr_image, df, e1, out):
        print("进入绘制图表")
        Width = self.flowrate_cam.Width.get()
        Height = self.flowrate_cam.Height.get()
        pred_seg = self.seg(bgr_image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bgr_image, "plan1", (50, 50), font, 1, (0, 0, 255), 2)

        cv2.putText(bgr_image, "Speed v: " + str(round(self.v_t / 2.7, 1)) + "m/s", (50, 100), font, 1,
                    (0, 255, 0), 2)  # 比例记得删 ！！！！！！！！！！！！！！！！！！！！
        self.flowrate_plot_qchart.y = round(self.v_t / 2.7, 1)
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
            df.to_csv("result/realtime_speed.csv", index=False)

        self.frame_idx += 1
        out.write(bgr_image)

        # 在大窗口中显示原视频
        pixmap = self.CvMatToQImage(bgr_image)
        ratio = max(Width / self.flowrate_Cam.width(), Height / self.flowrate_Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.flowrate_Cam.setPixmap(pixmap)

        # 在小窗口中显示ROI
        seg_pixmap = self.CvMatToQImage(pred_seg)
        roi_ratio = max(Width / self.flowrate_Cam.width(), Height / self.flowrate_Cam.height())
        seg_pixmap.setDevicePixelRatio(roi_ratio)
        self.flowrate_Cam_Seg.setPixmap(seg_pixmap)
        cv2.waitKey(1)

    def measurement(self, method="Trad"):
        # 保存视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("save_video/" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".mp4", fourcc,
                              24, (self.flowrate_cam.Width.get(), self.flowrate_cam.Height.get()))
        # 保存流速数据
        df = pd.DataFrame(columns=["时间", "流速v"])

        self.flowrate_choose_roi.setEnabled(False)
        self.flowrate_single.setEnabled(False)
        self.flowrate_continuous.setEnabled(False)

        self.flowrate_plot_qchart.timer.start()  # 开始曲线

        self.ref_precomp = xfeat.detectAndCompute(self.ref_frame, top_k=1024)[0]
        e1 = cv2.getTickCount()
        try:
            self.flowWorker.image_ready.disconnect()
        except TypeError:
            pass
        try:
            self.flowWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.flowCaculateWorker.done_prev_gray.disconnect()
        except TypeError:
            pass
        self.flowWorker.image_ready.connect(lambda bgr_image: self.flowCaculateWorker.serial_prev_gray(bgr_image))
        self.flowWorker.start_capture.connect(self.flowWorker.bgrimage_capture)
        self.flowCaculateWorker.done_prev_gray.connect(lambda: self.flowLoopWorker.flow_caculate(method, out, df, e1, fourcc))
        self.flowWorker.start_capture.emit()




    def crosssect_view(self, bgr_image: ndarray):

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        qimg = QImage(
            rgb_image.data, width, height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimg)

        cam_w = self.crosssect_Cam.width()
        cam_h = self.crosssect_Cam.height()
        ratio = max(width / cam_w, height / cam_h)
        pixmap.setDevicePixelRatio(ratio)

        self.crosssect_Cam.setPixmap(pixmap)

    def crosssect_plt(self, area:float):
        elapsed = time.time() - self._start_ts
        self.crosssect_plot_qchart.series.append(elapsed, area)
        # 滚动 X 轴，保留最近 10s
        self.crosssect_plot_qchart.axisX.setRange(max(0, elapsed - 10), elapsed)
        if area > self.crosssect_plot_qchart.axisY.max():
            self.crosssect_plot_qchart.axisY.setMax(area * 1.1)
    def _onCalcTimeout(self):

        # raw = self.crosssect_cam.data_stream[0].get_image()

        if self._test_index >= len(self._test_images):
            self._calc_timer.stop()
            return

            # —— 下面这几行替代原来从相机抓图的那一行 ——
        img_path = self._test_images[self._test_index]
        bgr = cv2.imread(img_path)  # 直接读 BGR 格式
        self._test_index += 1

        # 计算截面积
        area = van(bgr)

        elapsed = time.time() - self._start_ts
        # 更新曲线
        self.crosssect_plot_qchart.series.append(elapsed, area)
        # 滚动 X 轴，保留最近 10s
        self.crosssect_plot_qchart.axisX.setRange(max(0, elapsed - 10), elapsed)
        if area > self.crosssect_plot_qchart.axisY.max():
            self.crosssect_plot_qchart.axisY.setMax(area * 1.1)

    def seg(self, frame_bgr):
        """
        :param frame_rgb: Input RGB Image
        :return: Predict Segmentation
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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


class crosssect_QChartViewPlot(QChart):
    def __init__(self, parent=None):
        super(crosssect_QChartViewPlot, self).__init__()
        self.series = QSplineSeries()
        self.series.setName("截面积/cm^2")
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
        self.axisX.setRange(0, 10)
        self.axisY.setRange(0, 200)

        # self.timer.start()

    def handleTimeout(self):
        x = self.plotArea().width() / self.axisX.tickCount()
        y = (self.axisX.max() - self.axisX.min()) / self.axisX.tickCount()
        self.x += y
        self.series.append(self.x, self.y)
        self.scroll(x, 0)


class flowrate_QChartViewPlot(QChart):
    def __init__(self, parent=None):
        super(flowrate_QChartViewPlot, self).__init__()
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
    with open('design.qss', 'r', encoding='utf-8') as f:
        app.setStyleSheet(f.read())
    splash = SplashWithLog()
    splash.show()
    app.processEvents()
    do_initialization()
    MainWindow = QMainWindow()  # 主窗口
    w = MainForm(MainWindow)
    splash.close()
    MainWindow.show()
    sys.exit(app.exec_())
