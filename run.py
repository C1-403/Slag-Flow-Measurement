
import math
import os
import sys
import time
import pandas as pd
import glob
from PyQt5.QtChart import QDateTimeAxis
from PyQt5.QtCore import QDateTime
from collections import deque
import albumentations as AT
from datetime import  datetime
import cv2
import numpy as np
import keyboard
from PIL import Image
import matplotlib.dates as mdates
from pathlib import Path
import json
import torch
from matplotlib.widgets import SpanSelector
from matplotlib.font_manager import FontProperties
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtChart import QChartView, QChart, QValueAxis, QSplineSeries
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from albumentations.pytorch import ToTensorV2
import logging
import gxipy as gx
from configs import MyConfig
import matplotlib.ticker as mticker
from mainwindows import Ui_mainWindow
from SimplifiedMainwindow import Simplified_MainWindow,LoginWindow
from utils.splash.splash_ui import Ui_Form
from models import get_model
from modules.xfeat import XFeat
from utils import get_colormap, transforms
from statsmodels.tsa.arima.model import ARIMA
from numpy import ndarray
from van import *

from area_calculator import *
import warnings

import time

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QMessageBox
)

import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# 屏蔽所有 DeprecationWarning（包括 sipPyTypeDict 的弃用警告）
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 屏蔽所有 FutureWarning（包括 torch.load 的未来行为警告）
warnings.filterwarnings("ignore", category=FutureWarning)
def now_str_ms():
    """返回毫秒级时间字符串，例如 2026-04-22 14:35:12.123"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
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

class DateSelectDialog(QDialog):
    def __init__(self, dates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择日期")
        self.resize(300, 500)

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.addItems(dates)
        layout.addWidget(self.list_widget)

        self.btn_ok = QPushButton("查看当天曲线")
        self.btn_ok.clicked.connect(self.accept)
        layout.addWidget(self.btn_ok)

    def selected_date(self):
        item = self.list_widget.currentItem()
        return item.text() if item else None
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)
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
    ckpt_pth = "D:\\edge_download\\Slag-Flow-Measurement\\save\\best_fastscnn.pth"
    checkpoint = torch.load(ckpt_pth, map_location=torch.device(device))
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
            self.main.flowrate_cam.TriggerSoftware.send_command()
            if self.main.flowrate_cam is None:
                return  # 相机对象不存在，直接退出

            if not hasattr(self.main.flowrate_cam, "data_stream"):
                return  # 没有数据流属性，说明相机已关闭

            if len(self.main.flowrate_cam.data_stream) == 0:
                return
            raw_image = self.main.flowrate_cam.data_stream[0].get_image()
            Width = self.main.flowrate_cam.Width.get()
            Height = self.main.flowrate_cam.Height.get()
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()

            # 逆时针旋转 90 度
            numpy_image = cv2.rotate(numpy_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # 如果你要顺时针 90 度，就改成：
            # numpy_image = cv2.rotate(numpy_image, cv2.ROTATE_90_CLOCKWISE)

            h, w, ch = numpy_image.shape
            bytes_per_line = ch * w
            qimg = QImage(numpy_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            self.pixmap_ready.emit(pixmap)

    @pyqtSlot()
    def bgrimage_capture(self):
        if self.main.flowrate_cam_state:

            # 采集 + 转成 QPixmap ——
            self.main.flowrate_cam.stream_on()
            self.main.flowrate_cam.TriggerSoftware.send_command()
            if self.main.flowrate_cam is None:
                return  # 相机对象不存在，直接退出

            if not hasattr(self.main.flowrate_cam, "data_stream"):
                return  # 没有数据流属性，说明相机已关闭

            if len(self.main.flowrate_cam.data_stream) == 0:
                return
            raw_image = self.main.flowrate_cam.data_stream[0].get_image()
            if raw_image is not None:
                rgb_image = raw_image.convert("RGB")
                if rgb_image is not None:
                    numpy_image = rgb_image.get_numpy_array()
                    if numpy_image is not None and numpy_image.size > 0:
                        # 先旋转 90 度
                        # numpy_image = cv2.rotate(numpy_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        # # 如果要顺时针 90 度：
                        # # numpy_image = cv2.rotate(numpy_image, cv2.ROTATE_90_CLOCKWISE)

                        bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                        self.image_ready.emit(bgr_image)

    @pyqtSlot()
    def reset(self):
        """清空采集缓存"""
        # 如果有正在处理的 pixmap 或 image 引用，可以丢掉
        try:
            self.image_ready.disconnect()
        except TypeError:
            pass
        try:
            self.pixmap_ready.disconnect()
        except TypeError:
            pass


class FlowrateLoopWorker(QObject):
    """运行 flowrate 相关方法的 worker"""


    def __init__(self, main):
        super().__init__()
        self.main = main
        self._running = False
        self._timer = None  # 持久的定[object Object]器

        # 存储flow_caculate所需参数
        self._method = None
        self._last_start_time = 0

    @pyqtSlot(str, int)
    def start(self, method, interval_ms):
        if self._running:
            return
        self._running = True
        self._method = method
        self._last_start_time = time.time()

        try:
            self.main.flowWorker.image_ready.disconnect()
        except TypeError:
            pass
        try:
            self.main.flowWorker.start_capture.disconnect()
        except TypeError:
            pass

        self.main.flowWorker.image_ready.connect(
            lambda img: self.serial_measurement(img, self._method),
            Qt.QueuedConnection
        )
        self.main.flowWorker.start_capture.connect(
            self.main.flowWorker.bgrimage_capture,
            Qt.QueuedConnection
        )

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.flow_caculate)
        self._timer.start(interval_ms)


    @pyqtSlot()
    def flow_caculate(self):
        if not self._running or not self.main.flowrate_cam_state:
            return

        try:
            # —— 1. 切片文件、分段保存视频这些维护 ——
            now = time.time()
            if now - self._last_start_time >= self.main.duration:
                self.main.out.release()
                self.main.out = cv2.VideoWriter(
                    "save_video/" + now_str_ms() + ".mp4",
                    self.main.fourcc, 24,
                    (self.main.flowrate_cam.Width.get(), self.main.flowrate_cam.Height.get())
                )
                self._last_start_time = now
            #每帧时间开始记录
            date = datetime.now()
            timestamp = date.timestamp()
            self.main.curr_time = timestamp
            # —— 2. 发出一次采集信号 ——
            if self.main.flowrate_cam_state:
                self.main.flowWorker.start_capture.emit()

        except Exception as ex:
            logging.exception("FlowrateLoopWorker 内部出错：%s", ex)


    @pyqtSlot()
    def serial_measurement(self, bgr_image: ndarray, method):
        self.main.frame_counter += 1
        # if self.main.uimode == 0:
        #     with open("coor.json", "r") as file:
        #         data = json.load(file)
        #         coor = np.array(data)
        with open(resource_path("coor.json"), "r", encoding="utf-8") as file:
            data = json.load(file)
            coor = np.array(data)

        output = output_choose_vedio(coor, bgr_image)
        vis = output.copy()
        cv2.rectangle(bgr_image, tuple(coor[-2, :] - 1), tuple(coor[-1, :]), (0, 255, 0), 2)  # 在原视频显示选定的框的范围
        # 传统角点检测
        if method == "Trad":
            if self.main.uimode == 1:
                self.main.flowrate_xFeat_start.setEnabled(False)

            # curr_time = self.cam.get(cv2.CAP_PROP_POS_MSEC)  # 读取时间戳，用于计算单帧时间
            try:
                self.main.flowCaculateWorker.start_caculate.disconnect()
            except TypeError:
                pass
            try:
                self.main.flowCaculateWorker.done_caculate.disconnect()
            except TypeError:
                pass
            self.main.flowCaculateWorker.start_caculate.connect(
                lambda: self.main.flowCaculateWorker.trad_caculate(output, vis))
            self.main.flowCaculateWorker.done_caculate.connect(lambda: self.main.flowViewerWorker.caculate_view(bgr_image))
            self.main.flowCaculateWorker.start_caculate.emit()


        if method == "xFeat":
            if self.main.uimode == 1:
                self.main.flowrate_Trad_start.setEnabled(False)
            try:
                self.main.flowCaculateWorker.start_caculate.disconnect()
            except TypeError:
                pass
            try:
                self.main.flowCaculateWorker.done_caculate.disconnect()
            except TypeError:
                pass
            self.main.flowCaculateWorker.start_caculate.connect(
                lambda: self.main.flowCaculateWorker.xFeat_caculate(output, vis))
            self.main.flowCaculateWorker.done_caculate.connect(lambda: self.main.flowViewerWorker.caculate_view(bgr_image))
            self.main.flowCaculateWorker.start_caculate.emit()

    @pyqtSlot()
    def reset(self):
        """重置循环线程的内部状态"""
        self._running = False
        self._method = None
        self._last_start_time = 0
        if self._timer:
            self._timer.stop()
            self._timer = None
        # 如果后续加了单帧缓冲变量，这里也清空
        if hasattr(self, "_latest_frame"):
            self._latest_frame = None
        if hasattr(self, "_busy"):
            self._busy = False

class FlowrateViewerWorker(QObject):
    """运行 flowrate 相关方法的 worker"""
    done_viewer = pyqtSignal()
    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def flowrate_Camera(self, pixmap: QPixmap):
        """
        接收到子线程传来的 QPixmap，用于更新 UI
        """
        # 获取图像的宽高比，使其适应显示窗口
        Width = self.main.flowrate_cam.Width.get()
        Height = self.main.flowrate_cam.Height.get()
        ratio = max(Width / self.main.flowrate_Cam.width(), Height / self.main.flowrate_Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.main.flowrate_Cam.setPixmap(pixmap)

    @pyqtSlot()
    def serial_flowrate_Choose_ROI(self, bgr_image: ndarray):
        Width = self.main.flowrate_cam.Width.get()
        Height = self.main.flowrate_cam.Height.get()
        get_choose_action(bgr_image, OnMouseAction)
        self.main.ref_frame = output_choose_vedio(coor, bgr_image)
        self.main.Width_choose = coor[-1, 0] - coor[-2, 0]  # 选中区域的宽
        self.main.Height_choose = coor[-1, 1] - coor[-2, 1]  # 选中区域的高
        print("视频选中区域的宽：%d" % self.main.Width_choose, '\n'"视频选中区域的高：%d" % self.main.Height_choose)

        cv2.rectangle(bgr_image, tuple(coor[-2, :] - 1), tuple(coor[-1, :]), (0, 255, 0), 2)  # 在原图像显示选定的框的范围
        # 将opencv格式转为QImage，在主窗口内显示原图像
        pixmap = self.main.CvMatToQImage(bgr_image)
        ratio = max(Width / self.main.flowrate_Cam.width(), Height / self.main.flowrate_Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.main.flowrate_Cam.setPixmap(pixmap)
        if self.main.uimode == 1:
            self.main.flowrate_Trad_start.setEnabled(True)
            self.main.flowrate_xFeat_start.setEnabled(True)
        try:
            with open(resource_path("coor.json"), "w", encoding="utf-8") as file:
                json.dump(coor.tolist(), file)  # 将 NumPy 数组转为列表后保存
            print("Coordinates saved to 'coor.json'")
        except Exception as e:
            print(f"Error saving coordinates to file: {e}")

    @pyqtSlot()
    def caculate_view(self, bgr_image):
        Width = self.main.flowrate_cam.Width.get()
        Height = self.main.flowrate_cam.Height.get()
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.main.uimode == 1:
            pred_seg = self.main.seg(bgr_image)
            cv2.putText(pred_seg, "Flow rate ROI area", (50, 50), font, 1, (0, 0, 255), 2)
        cv2.putText(bgr_image, "The flow velocity camera captures video frames.", (50, 50), font, 1, (0, 0, 255), 2)
        cv2.putText(bgr_image, "Speed v: " + str(round(self.main.v_t , 1)) + "m/s", (50, 100), font, 1,
                    (0, 255, 0), 2)


        # self.main.plot_qchart.x = curr_time
        cv2.putText(bgr_image, "Operating_time: " + str(round(self.main.cost_time, 3)) + "ms", (50, 150), font, 1,
                    (0, 255, 0), 2)
        # if self.main.uimode == 0:
        #     try:
        #         self.main.flowViewerWorker.done_viewer.disconnect()
        #     except TypeError:
        #         pass
        #     self.main.flowViewerWorker.done_viewer.connect(
        #         self.main.trafficWorker.traffic_caculate)
        #     self.main.flowViewerWorker.done_viewer.emit()
        #
        #
        #     # ARIMA预测
        #     def Kar(measurement, prediction, alpha=0.5):
        #         """
        #         卡尔曼滤波器
        #         measurement: 真实测量值
        #         prediction: 预测值（采用ARIMA预测值）
        #         return: 卡尔曼滤波器预测结果
        #         """
        #         result = (1 - alpha) * measurement + alpha * prediction
        #         return result
        #     if len(self.main.v_history) >= 400:
        #         data = list(self.main.v_history)
        #         start = time.time()
        #         sub_data = data[:400]
        #         # draw_data(sub_data)
        #         model = ARIMA(sub_data, order=(1, 1, 1)).fit()  # 构建ARIMA模型，order参数表示p、d、q
        #         predict_data = model.predict(0, 430)  # 预测数据
        #         forecast = model.forecast(30)  # 预测未来数据
        #         end = time.time()
        #
        #         # 进行卡尔曼滤波
        #         result_data = []
        #         for i in range(1, 400):
        #             result_data.append(Kar(sub_data[i - 1], predict_data[i], alpha=0.8))
        #         print(end - start)
        #
        #         # 更新图表
        #         self.main.arima_plot_qchart.update_chart(sub_data, predict_data, result_data)


        # 图表数据
        if self.main.frame_idx % self.main.iters == 0 and self.main.frame_idx != 0:
            self.main.e2 = cv2.getTickCount()
            c_time = (self.main.e2 - self.main.e1) / cv2.getTickFrequency()
            self.main.cost_time = c_time * 1000 / self.main.iters
            self.main.e1 = self.main.e2

            now_str = now_str_ms()
            self.main.speeds.append(self.main.v_t)
            self.main.frames.append(now_str)
            self.main.flowrate_plot_qchart.update_value(self.main.v_t)
            row = {
                "时间": now_str,
                "流速v": round(self.main.v_t, 4)
            }
            self.main.flowrate_buffer.append(row)

            # 每隔 flush_interval 秒批量写一次 CSV
            now_ts = time.time()
            if now_ts - self.main.last_flowrate_flush >= self.main.flush_interval:
                self.main.append_rows_to_csv(
                    "result/realtime_flowrate.csv",
                    self.main.flowrate_buffer,
                    ["时间", "流速v"]
                )
                self.main.flowrate_buffer.clear()
                self.main.last_flowrate_flush = now_ts



        self.main.frame_idx += 1
        self.main.out.write(bgr_image)

        # 在大窗口中显示原视频
        if self.main.uimode == 1:
            pixmap = self.main.CvMatToQImage(bgr_image)
            ratio = max(Width / self.main.flowrate_Cam.width(), Height / self.main.flowrate_Cam.height())
            pixmap.setDevicePixelRatio(ratio)
            self.main.flowrate_Cam.setPixmap(pixmap)


        # 在小窗口中显示ROI
        if self.main.uimode == 1:
            seg_pixmap = self.main.CvMatToQImage(pred_seg)
            roi_ratio = max(Width / self.main.flowrate_Cam.width(), Height / self.main.flowrate_Cam.height())
            seg_pixmap.setDevicePixelRatio(roi_ratio)
            self.main.flowrate_Cam_Seg.setPixmap(seg_pixmap)
        # ===== 简化界面下，显示当前采集流速画面 + 数值 =====
        if self.main.uimode == 0:
            self.main.update_simple_camera_view(bgr_image)
            self.main.update_simple_values()
            #self.main.check_auto_update_bottom_contour()




    @pyqtSlot()
    def reset(self):
        """清空显示状态"""
        try:
            self.main.flowrate_Cam.clear()
            self.main.flowrate_Cam_Seg.clear()
        except Exception:
            pass



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
        # if self.main.uimode == 0:
        #     with open("coor.json", "r") as file:
        #         data = json.load(file)
        #         coor = np.array(data)
        with open(resource_path("coor.json"), "r", encoding="utf-8") as file:
            data = json.load(file)
            coor = np.array(data)
        output = output_choose_vedio(coor, bgr_image)
        self.main.ref_frame = output
        self.main.ref_precomp = xfeat.detectAndCompute(self.main.ref_frame, top_k=1024)[0]
        # 滤波+锐化
        frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
        frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
        frame_gray = cv2.filter2D(frame_gray, -1, kernel)
        self.main.prev_gray = frame_gray
        self.done_prev_gray.emit()
        print("进入滤波计算")

    def is_near_black(self, bgr_image: ndarray) -> bool:
        """
        判断ROI是否近乎全黑：
        1) 平均灰度很低
        2) 大部分像素都接近黑色
        两个条件同时满足时，认为当前画面几乎全黑
        """
        if bgr_image is None or bgr_image.size == 0:
            return True

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        mean_val = float(np.mean(gray))
        black_ratio = float(np.mean(gray < self.main.black_pixel_threshold))

        print(f"[黑场检测] mean={mean_val:.2f}, black_ratio={black_ratio:.3f}")

        return (
                mean_val < self.main.black_mean_threshold
                and black_ratio > self.main.black_ratio_threshold
        )
    @pyqtSlot()
    def trad_caculate(self, output, vis):

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
                print(f"当前帧开始时间{self.main.curr_time}，上一帧开始时间{self.main.prev_time}")
                self.main.v = self.main.d_ave / (self.main.curr_time - self.main.prev_time)
                self.main.v_sum += self.main.v

        if self.main.frame_idx % self.main.detect_interval == 0:  # 每几帧检测一次特征点
            mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
            mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
            for x, y in [np.int32(tr[-1]) for tr in self.main.tracks]:  # 跟踪的角点画圆
                cv2.circle(mask, (x, y), 5, 0, -1)

        # 计算平均速度
        if self.main.frame_idx % self.main.iters == 0 and self.main.frame_idx != 0 and self.main.curr_time - self.main.prev_time != 0:
            self.main.v_t = self.main.v_sum / (self.main.num + 0.00001)  # 避免除0
            self.main.v_t = round(self.main.v_t, 6) * self.main.transform * 1000
            self.main.v_sum = 0
            self.main.num = 0
            # 保存历史速度
            self.main.v_history.append(self.main.v_t)
        # Shi-Tomasi角点检测
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.main.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中
        self.main.prev_gray = frame_gray
        self.main.prev_time = self.main.curr_time

        self.done_caculate.emit()

    @pyqtSlot()
    def xFeat_caculate(self, output, vis):
        self.main.stop_get = False
        if self.is_near_black(output):
            print("检测到ROI近乎全黑，速度直接置零，跳过xFeat匹配")
            self.main.stop_get = True
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
            temp = math.atan2(y1 - y2, x1 - x2) / math.pi * 180
            dis = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))
            x_dis = dis * self.main.transform * 1000
            if self.main.angle - 10 < temp < self.main.angle + 10 and dis > 5 and x_dis < 30:  # 流动方向和速度筛选
                new_pt1.append((x1, y1, dis))
                new_pt2.append((x2, y2))
        self.main.d_sum = 0
        self.main.num = 0
        if len(new_pt1) > 10:
            for (x1, y1, dis), (x2, y2) in zip(new_pt1, new_pt2):
                cv2.circle(vis, (int(x1), int(y1)), 2, (0, 255, 0), -1)
                track = np.array([[x1, y1], [x2, y2]], np.int32)
                cv2.polylines(vis, [track], True, (0, 255, 0))
                x_dis = dis * self.main.transform * 1000

                self.main.d_sum = self.main.d_sum + dis
                self.main.num = self.main.num + 1
                # print("d_sum:"+str(self.main.d_sum))

        #cv2.imshow('vis', vis)
        self.main.vis_view = vis
        dt = self.main.curr_time - self.main.prev_time

        if (
                self.main.frame_idx % self.main.iters == 0
                and self.main.frame_idx != 0
                and self.main.num > 0
                and dt > 0
        ):
            self.main.d_ave  = self.main.d_sum / (self.main.num + 0.00001)

            self.main.v = self.main.d_ave / dt
            raw_v_t = round(self.main.v, 6) * self.main.transform * 1000 / 2.2
            print("raw_v_t",raw_v_t)
            print("dt:",dt)
            valid = True
            # 1) 绝对范围过滤
            # if raw_v_t > self.main.xfeat_abs_max_speed:
            #     valid = False
            #     print("xFeat速度异常：超过绝对阈值，丢弃")
            #
            # # 2) 时间间隔过滤（卡顿导致dt异常）
            # if dt < self.main.xfeat_min_dt or dt > self.main.xfeat_max_dt :
            #     # 因为你的 curr_time = timestamp * 100，所以这里乘100
            #     valid = False
            #     print("xFeat速度异常：时间间隔异常，丢弃")
            #
            # # 3) 相对跳变过滤
            # if valid and len(self.main.xfeat_speed_buffer) >= 3:
            #     mean_v = sum(self.main.xfeat_speed_buffer) / len(self.main.xfeat_speed_buffer)
            #
            #     if mean_v > 1e-6:
            #         jump_ratio = abs(raw_v_t - mean_v) / mean_v
            #         if jump_ratio > self.main.xfeat_max_jump_ratio:
            #             valid = False
            #             print(f"xFeat速度异常：跳变过大 jump_ratio={jump_ratio:.3f}，丢弃")

            # 4) 合法值进入缓冲，否则沿用旧值
            if valid:
                self.main.xfeat_speed_buffer.append(raw_v_t)
                self.main.xfeat_last_valid_v = raw_v_t

                # 用缓冲均值作为最终输出，更稳定
                self.main.v_t = sum(self.main.xfeat_speed_buffer) / len(self.main.xfeat_speed_buffer)

                print("流速计算结果(滤波后)", self.main.v_t)
                self.main.v_history.append(self.main.v_t)
            else:
                # 当前值异常，不更新缓冲，保持上一有效值
                if len(self.main.xfeat_speed_buffer) > 0:
                    self.main.v_t = sum(self.main.xfeat_speed_buffer) / len(self.main.xfeat_speed_buffer)
                else:
                    self.main.v_t = self.main.xfeat_last_valid_v

                print("沿用上一稳定流速", self.main.v_t)
            if self.is_near_black(output):
                print("检测到ROI近乎全黑，速度直接置零，跳过xFeat匹配")
                self.main.stop_get = True
                self.main.v = 0.0
                self.main.v_t = 0.0



        self.main.prev_time = self.main.curr_time
        self.main.ref_precomp = current
        self.done_caculate.emit()


class CrossCaptureWorker(QObject):
    """运行 crosssect 相关方法的 worker"""
    start_capture = pyqtSignal()
    done_prev_gray = pyqtSignal()
    image_ready = pyqtSignal(ndarray)
    pixmap_ready = pyqtSignal(QPixmap)

    def __init__(self, main):
        super().__init__()
        self.main = main
        # 调试图目录
        self.debug_dir = Path(r'D:\edge_download\Slag-Flow-Measurement\cross_full')

        # 当前读取到第几张
        self.debug_index = 0
    @pyqtSlot()
    def process_cross(self):
        if self.main.crosssect_cam_state:
            # 采集 + 转成 QPixmap ——
            self.main.crosssect_cam.stream_on()
            self.main.crosssect_cam.TriggerSoftware.send_command()
            raw_image = self.main.crosssect_cam.data_stream[0].get_image()
            Width = self.main.crosssect_cam.Width.get()
            Height = self.main.crosssect_cam.Height.get()
            # raw_image = cv2.resize(raw_image, (Width, Height))
            if raw_image != None:
                rgb_image = raw_image.convert("RGB")
                if rgb_image != None:
                    numpy_image = rgb_image.get_numpy_array()
                    if numpy_image is not None and numpy_image.size > 0:
                        pixmap = QImage(numpy_image, Width, Height, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(pixmap)
                        self.pixmap_ready.emit(pixmap)

    # @pyqtSlot()
    # def bgrimage_capture(self):
    #     if not self.main.crosssect_cam_state or self.main.crosssect_cam is None:
    #         return
    #
    #     if not hasattr(self.main.crosssect_cam, "data_stream"):
    #         return
    #
    #     if len(self.main.crosssect_cam.data_stream) == 0:
    #         return
    #     self.main.crosssect_cam.stream_on()
    #     self.main.crosssect_cam.TriggerSoftware.send_command()
    #     raw_image = self.main.crosssect_cam.data_stream[0].get_image()
    #     if raw_image is None:
    #         return
    #
    #     rgb_image = raw_image.convert("RGB")
    #     if rgb_image is None:
    #         return
    #
    #     numpy_image = rgb_image.get_numpy_array()
    #     if numpy_image is None or numpy_image.size == 0:
    #         return
    #
    #     bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    #     # 保存最近一帧截面积图像，供简化界面自动更新下表面轮廓使用
    #     self.main.latest_cross_bgr = bgr_image.copy()
    #
    #     # 直接发内存图，不再写盘再读盘
    #     self.image_ready.emit(bgr_image.copy())

    @pyqtSlot()
    def bgrimage_capture(self):
        if not self.debug_dir.exists() or not self.debug_dir.is_dir():
            print(f"调试目录不存在: {self.debug_dir}")
            return

        # 获取目录下所有 png，按文件名排序
        png_files = sorted(self.debug_dir.glob("*.png"))
        if not png_files:
            print(f"目录下没有 png 图像: {self.debug_dir}")
            return

        # 循环读取：超出后从头开始
        img_path = png_files[self.debug_index]
        self.debug_index = (self.debug_index + 1) % len(png_files)

        bgr_image = cv2.imread(str(img_path))
        if bgr_image is None:
            print(f"读取调试图像失败: {img_path}")
            return

        print(f"当前读取图像: {img_path.name}")

        # 保存最近一帧截面积图像，供简化界面自动更新下表面轮廓使用
        self.main.latest_cross_bgr = bgr_image.copy()

        # 直接发内存图
        self.image_ready.emit(bgr_image.copy())

class CrossCaculateWorker(QObject):
    """运行 crosssect 相关方法的 worker"""

    done_caculate = pyqtSignal(float)


    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def single_crosssect_caculate(self, bgr_image: ndarray):
        if self.main.stop_get == False:
            area = abs(van(bgr_image))

            if area is not None :
                print("area",area)
                self.done_caculate.emit(area)
                self.main.area = area

                # 保存历史截面积
                self.main.area_history.append(self.main.area)
    @pyqtSlot()
    def reset(self):
        """重置计算线程状态"""
        try:
            self.done_caculate.disconnect()
        except TypeError:
            pass


class CrossViewerWorker(QObject):
    """运行 crosssect 相关方法的 worker"""


    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot()
    def cross_Camera(self, pixmap: QPixmap):
        """
		接收到子线程传来的 QPixmap，用于更新 UI
		"""
        # 获取图像的宽高比，使其适应显示窗口
        Width = self.main.crosssect_cam.Width.get()
        Height = self.main.crosssect_cam.Height.get()
        ratio = max(Width / self.main.crosssect_Cam.width(), Height / self.main.crosssect_Cam.height())
        pixmap.setDevicePixelRatio(ratio)
        self.main.crosssect_Cam.setPixmap(pixmap)
    @pyqtSlot()
    def crosssect_plt(self, area: float):

        # 更新 DataFrame 并保存截面积数据
        now_str = now_str_ms()
        print(3)
        self.main.crosssect_areas.append(area)
        self.main.crosssect_times.append(now_str)
        self.main.crosssect_plot_qchart.update_value(area)
        row = {
            "时间": now_str,
            "截面积": round(area, 4)
        }
        self.main.crosssect_buffer.append(row)

        now_ts = time.time()
        if now_ts - self.main.last_crosssect_flush >= self.main.flush_interval:
            self.main.append_rows_to_csv(
                "result/realtime_crosssect_area.csv",
                self.main.crosssect_buffer,
                ["时间", "截面积"]
            )
            self.main.crosssect_buffer.clear()
            self.main.last_crosssect_flush = now_ts



    @pyqtSlot()
    def crosssect_view(self, bgr_image: ndarray):
        if self.main.uimode == 1:

            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_image, "cross-sectional area calculation frame",
                        (50, 50), font, 1, (255, 0, 0), 2)

            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            qimg = QImage(
                rgb_image.data, width, height,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qimg)

            cam_w = self.main.crosssect_Cam.width()
            cam_h = self.main.crosssect_Cam.height()
            ratio = max(width / cam_w, height / cam_h)
            pixmap.setDevicePixelRatio(ratio)

            self.main.crosssect_Cam.setPixmap(pixmap)

    @pyqtSlot()
    def reset(self):
        """重置视图显示"""
        # 清空图表
        if hasattr(self.main.crosssect_plot_qchart, "series"):
            self.main.crosssect_plot_qchart.series.clear()

        # 清空相机显示
        if self.main.uimode == 1:
            if hasattr(self.main.crosssect_Cam, "clear"):
                self.main.crosssect_Cam.clear()

class CrossLoopWorker(QObject):
    """运行 crosssect 相关方法的 worker"""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._running = False
        self._timer = None  # 持久的定[object Object]器

    @pyqtSlot(int)
    def start(self,interval_ms):
        self.main.crosssect_cam_state = True
        if self.main.uimode == 1:
            self.main.crosssect_continuous.setEnabled(False)
            self.main.crosssect_CloseCam.setEnabled(True)
            self.main.crosssect_OpenCam.setEnabled(False)
            self.main.crosssect_start.setEnabled(False)
            self.main.bottom_face_start.setEnabled(False)
            self.main.crosssectcamera_param_control.setEnabled(False)
        self.main._start_ts = time.time()
        if self._running:
            return
        self._running = True

        try:
            self.main.CrossWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.main.CrossWorker.image_ready.disconnect()
        except TypeError:
            pass
        try:
            self.main.CrossCaculateWorker.done_caculate.disconnect()
        except TypeError:
            pass
        if self.main.crosssect_cam is not None:
            self.main.CrossWorker.start_capture.connect(self.main.CrossWorker.bgrimage_capture,
                Qt.QueuedConnection)
            self.main.CrossWorker.image_ready.connect(
                lambda bgr_image: self.main.CrossCaculateWorker.single_crosssect_caculate(bgr_image),
                Qt.QueuedConnection)
            if self.main.uimode == 1:
                self.main.CrossWorker.image_ready.connect(lambda bgr_image: self.main.CrossViewerWorker.crosssect_view(bgr_image),
                    Qt.QueuedConnection)
            self.main.CrossCaculateWorker.done_caculate.connect(lambda area: self.main.CrossViewerWorker.crosssect_plt(area),
                Qt.QueuedConnection)

        self._timer = QTimer(self)
        self.main.crosssect_plot_qchart.timer.start()
        self._timer.timeout.connect(self.crosssect_caculate)
        self._timer.start(interval_ms)
    @pyqtSlot()
    def crosssect_caculate(self):
        if  self.main.crosssect_cam_state:
            self.main.CrossWorker.start_capture.emit()

    @pyqtSlot()
    def reset(self):
        """重置循环线程状态"""
        if self._timer:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None

        try:
            self.main.CrossWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.main.CrossWorker.image_ready.disconnect()
        except TypeError:
            pass
        try:
            self.main.CrossCaculateWorker.done_caculate.disconnect()
        except TypeError:
            pass

        self._running = False
        self.main.crosssect_cam_state = False
        self.main._start_ts = None

class BottomfaceWorker(QObject):
    """运行 crosssect 相关方法的 worker"""
    start_capture = pyqtSignal()
    def __init__(self, main):
        super().__init__()
        self.main = main


    @pyqtSlot()
    def Bottomface_caculate(self):
        if self.main.uimode == 1:
            self.main.bottom_face_start.setEnabled(False)
        self.main.crosssect_cam_state = True

        if self.main.crosssect_cam_state:
            try:
                self.main.BottomfaceWorker.start_capture.disconnect()
            except TypeError:
                pass
            try:
                self.main.CrossWorker.image_ready.disconnect()
            except TypeError:
                pass
            self.main.BottomfaceWorker.start_capture.connect(self.main.CrossWorker.bgrimage_capture)

            self.main.CrossWorker.image_ready.connect(lambda loaded_image: save_lower_countour(loaded_image,SAVE_ROI=True))
            self.main.BottomfaceWorker.start_capture.emit()
        else:
            print("相机关闭，无法继续计算下表面面积。")
        time.sleep(1)
        if self.main.uimode == 1:
            self.main.bottom_face_start.setEnabled(True)

class TrafficWorker(QObject):
    """运行 crosssect 相关方法的 worker"""

    def __init__(self, main):
        super().__init__()
        self.main = main
        # 历史瞬时流量 Q 队列（单位 m³/s）
        self.history_len = 10
        self.q_history = deque(maxlen=self.history_len)
        # 上一次调用 traffic_caculate 的时间戳
        self.last_time = None

    @pyqtSlot()
    def traffic_caculate(self):
        # ===== 1. 计算 dt =====
        now = time.time()
        if not hasattr(self, "last_time"):
            self.last_time = None

        if self.last_time is None:
            dt = 0  # 第一次调用
        else:
            dt = now - self.last_time
        self.last_time = now
        # ===== 2. 从 main 历史速度 & 截面积队列，计算历史流量序列 =====
        if len(self.main.v_history) < 2 or len(self.main.area_history) < 2:
            # 历史数据不足，用当前瞬时值
            m_Q = self.main.v_t * self.main.area
        else:
            # 把历史 v 和 area 转成 Q 序列
            n = min(len(self.main.v_history), len(self.main.area_history), 10)
            v_recent = list(self.main.v_history)[-n:]  # 最近 n 个 v
            area_recent = list(self.main.area_history)[-n:]  # 最近 n 个 area

            q_list = [
                v_recent[i] * area_recent[i]
                for i in range(n)
            ]
            # ===== 3. 多点梯形积分求平均流量（m³/s）=====
            if dt == 0:
                m_Q = q_list[-1]  # 防止除零
            else:
                # 多点梯形积分法，先算总体积，再除以总时间得平均流量
                total_volume = 0.0
                for i in range(len(q_list) - 1):
                    total_volume += (q_list[i] + q_list[i + 1]) / 2 * dt
                total_time = dt * (len(q_list) - 1)
                m_Q = total_volume / total_time
        # 5. 绘图
        # 6. 保存到 DataFrame
        self.main.traffic.append(m_Q)
        self.main.traffic_times.append(now_str_ms())
        # 将新行添加到 DataFrame
        now_str = now_str_ms()


        self.main.traffic.append(m_Q)
        self.main.traffic_times.append(now_str)

        row = {
            "时间": now_str,
            "流量": round(m_Q, 2)
        }
        self.main.traffic_buffer.append(row)
        if hasattr(self.main, "traffic_plot_qchart"):
            self.main.traffic_plot_qchart.update_value(m_Q)
        now_ts = time.time()
        if now_ts - self.main.last_traffic_flush >= self.main.flush_interval:
            self.main.append_rows_to_csv(
                "result/realtime_traffic.csv",
                self.main.traffic_buffer,
                ["时间", "流量"]
            )
            self.main.traffic_buffer.clear()
            self.main.last_traffic_flush = now_ts

class TrafficLoopWorker(QObject):
    """独立运行流量计算的循环 worker"""
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._running = False
        self._timer = None

    @pyqtSlot(int)
    def start(self, interval_ms):
        if self._running:
            return

        self._running = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.traffic_caculate_loop)
        self._timer.start(interval_ms)

    @pyqtSlot()
    def traffic_caculate_loop(self):
        if not self._running:
            return

        # 至少要求流速或截面积系统有一个在运行，否则没必要算
        if not self.main.flowrate_cam_state and not self.main.crosssect_cam_state:
            return

        try:
            self.main.trafficWorker.traffic_caculate()
        except Exception as e:
            logging.exception("TrafficLoopWorker 内部出错: %s", e)

    @pyqtSlot()
    def reset(self):
        self._running = False
        if self._timer:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
class HistoryPlotDialog(QDialog):
    def __init__(self, parent=None, title="历史流速/流量曲线", allow_day_select=False, date_list=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.allow_day_select = allow_day_select
        self.date_list = date_list or []

        self.setWindowFlag(Qt.Window, True)
        self.setModal(False)

        # ===== 字体 =====
        self.btn_font = QFont("Microsoft YaHei", 14, QFont.Bold)
        self.title_fontsize = 22
        self.label_fontsize = 18
        self.tick_fontsize = 15
        self.line_width = 1.0

        # ===== 数据 =====
        self.full_df_v_raw = pd.DataFrame()
        self.full_df_q_raw = pd.DataFrame()

        self.full_df_v = pd.DataFrame()
        self.full_df_q = pd.DataFrame()

        # 给按天查看保留原始数据
        self.df_v_all = pd.DataFrame()
        self.df_q_all = pd.DataFrame()

        self.title1 = "历史流速曲线"
        self.title2 = "历史质量流量曲线"

        self.current_xmin = None
        self.current_xmax = None

        self._plot_cache_v = {}
        self._plot_cache_q = {}
        self._data_version = 0
        self._events_connected = False

        main_layout = QVBoxLayout(self)

        # ---------- 顶部按钮栏 ----------
        top_bar = QHBoxLayout()

        self.btn_back = QPushButton("返回")
        self.btn_back.setMinimumHeight(42)
        self.btn_back.setMinimumWidth(100)
        self.btn_back.setFont(self.btn_font)
        self.btn_back.clicked.connect(self.handle_back)
        top_bar.addWidget(self.btn_back, alignment=Qt.AlignLeft)

        self.btn_reset = QPushButton("重置缩放")
        self.btn_reset.setMinimumHeight(42)
        self.btn_reset.setMinimumWidth(120)
        self.btn_reset.setFont(self.btn_font)
        self.btn_reset.clicked.connect(self.reset_zoom)
        top_bar.addWidget(self.btn_reset, alignment=Qt.AlignLeft)

        self.btn_save = QPushButton("保存曲线图")
        self.btn_save.setMinimumHeight(42)
        self.btn_save.setMinimumWidth(140)
        self.btn_save.setFont(self.btn_font)
        self.btn_save.clicked.connect(self.save_plot_png)
        top_bar.addWidget(self.btn_save, alignment=Qt.AlignLeft)

        if self.allow_day_select:
            self.btn_pick_day = QPushButton("选择日期")
            self.btn_pick_day.setMinimumHeight(42)
            self.btn_pick_day.setMinimumWidth(120)
            self.btn_pick_day.setFont(self.btn_font)
            self.btn_pick_day.clicked.connect(self.open_day_selector)
            top_bar.addWidget(self.btn_pick_day, alignment=Qt.AlignLeft)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # ---------- 画布 ----------
        self.figure = Figure(figsize=(16, 9))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(2, 1, 1)
        self.ax2 = self.figure.add_subplot(2, 1, 2, sharex=self.ax1)

        self.span1 = None
        self.span2 = None

    def save_plot_png(self):
        try:
            os.makedirs("result", exist_ok=True)
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            safe_title = self.windowTitle().replace(" ", "_").replace("/", "_").replace("\\", "_")
            file_path = os.path.join("result", f"{safe_title}_{now_str}.png")
            self.figure.savefig(file_path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "保存成功", f"曲线图已保存到：\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存曲线图失败：{e}")

    def handle_back(self):
        self.close()

    def reset_zoom(self):
        self.reset_full_view()

    def compress_zero_segments(self, df, value_col, zero_threshold=1e-8, keep_edge=2):
        """
        对连续零段做轻量压缩：
        - 非零点全部保留
        - 零段保留：段首、段尾、前后邻接点、以及少量边缘点
        避免零段被折线错误连成斜线
        """
        if df is None or df.empty:
            return df

        df = df.copy().reset_index(drop=True)
        is_zero = df[value_col].fillna(0).abs() <= zero_threshold

        keep_indices = set(df.index[~is_zero].tolist())
        n = len(df)
        start = None

        for i, flag in enumerate(is_zero):
            if flag and start is None:
                start = i
            elif (not flag) and (start is not None):
                end = i - 1

                for j in range(start, min(start + keep_edge, end + 1)):
                    keep_indices.add(j)
                for j in range(max(start, end - keep_edge + 1), end + 1):
                    keep_indices.add(j)

                if start - 1 >= 0:
                    keep_indices.add(start - 1)
                keep_indices.add(start)
                keep_indices.add(end)
                if end + 1 < n:
                    keep_indices.add(end + 1)

                start = None

        if start is not None:
            end = n - 1
            for j in range(start, min(start + keep_edge, end + 1)):
                keep_indices.add(j)
            for j in range(max(start, end - keep_edge + 1), end + 1):
                keep_indices.add(j)

            if start - 1 >= 0:
                keep_indices.add(start - 1)
            keep_indices.add(start)
            keep_indices.add(end)
            if end + 1 < n:
                keep_indices.add(end + 1)

        keep_indices = sorted(keep_indices)
        return df.loc[keep_indices].copy().reset_index(drop=True)

    def plot_history(self, df_v, df_q, compress_zero=True, thin_all_history=False,
                     title1="历史流速曲线", title2="历史质量流量曲线"):
        self.set_history_data(
            df_v=df_v,
            df_q=df_q,
            compress_zero=compress_zero,
            title1=title1,
            title2=title2
        )

    def downsample_minmax(self, df, x_col, y_col, target_points=2000):
        if df is None or df.empty:
            return df

        n = len(df)
        if n <= target_points:
            return df.reset_index(drop=True)

        bucket_size = int(np.ceil(n / target_points))
        y = df[y_col].to_numpy()
        keep_indices = set()

        for start in range(0, n, bucket_size):
            end = min(start + bucket_size, n)
            if end - start <= 0:
                continue

            chunk = y[start:end]
            keep_indices.add(start)
            keep_indices.add(end - 1)

            local_min = start + int(np.argmin(chunk))
            local_max = start + int(np.argmax(chunk))
            keep_indices.add(local_min)
            keep_indices.add(local_max)

            if len(chunk) >= 3:
                dy = np.abs(np.diff(chunk))
                local_change = start + int(np.argmax(dy))
                keep_indices.add(local_change)
                if local_change + 1 < end:
                    keep_indices.add(local_change + 1)

        keep_indices = np.array(sorted(i for i in keep_indices if 0 <= i < n))
        return df.iloc[keep_indices].copy().reset_index(drop=True)

    def get_target_points(self):
        """
        首屏更保守，优先保证打开快。
        """
        width_px = max(self.canvas.width(), 800)
        target = int(width_px * 1.2)
        return max(800, min(target, 2500))

    def _format_ts(self, x, pos=None):
        try:
            dt = mdates.num2date(x)
            return dt.strftime("%Y-%m-%d\n%H:%M:%S.%f")[:-3]
        except Exception:
            return ""

    def _make_cache_key(self, x_min, x_max, target_points, kind):
        return (
            self._data_version,
            kind,
            None if x_min is None else round(float(x_min), 3),
            None if x_max is None else round(float(x_max), 3),
            int(target_points)
        )

    def _ensure_events_connected(self):
        if self._events_connected:
            return

        from matplotlib.widgets import SpanSelector

        try:
            self.span1 = SpanSelector(
                self.ax1,
                self.on_select_xrange,
                "horizontal",
                useblit=True,
                interactive=True,
                props=dict(alpha=0.2, facecolor="tab:blue")
            )
            self.span2 = SpanSelector(
                self.ax2,
                self.on_select_xrange,
                "horizontal",
                useblit=True,
                interactive=True,
                props=dict(alpha=0.2, facecolor="tab:blue")
            )
        except TypeError:
            self.span1 = SpanSelector(
                self.ax1,
                self.on_select_xrange,
                "horizontal",
                useblit=True,
                rectprops=dict(alpha=0.2, facecolor="tab:blue")
            )
            self.span2 = SpanSelector(
                self.ax2,
                self.on_select_xrange,
                "horizontal",
                useblit=True,
                rectprops=dict(alpha=0.2, facecolor="tab:blue")
            )

        self._events_connected = True

    def set_history_data(self, df_v, df_q, compress_zero=True,
                         title1="历史流速曲线", title2="历史质量流量曲线"):
        self.title1 = title1
        self.title2 = title2

        self.full_df_v_raw = df_v.copy() if df_v is not None else pd.DataFrame()
        self.full_df_q_raw = df_q.copy() if df_q is not None else pd.DataFrame()

        self.df_v_all = self.full_df_v_raw.copy()
        self.df_q_all = self.full_df_q_raw.copy()

        if not self.full_df_v_raw.empty and "时间" in self.full_df_v_raw.columns:
            self.full_df_v_raw = self.full_df_v_raw.sort_values("时间").reset_index(drop=True)
            self.full_df_v_raw["ts"] = mdates.date2num(self.full_df_v_raw["时间"].dt.to_pydatetime())

        if not self.full_df_q_raw.empty and "时间" in self.full_df_q_raw.columns:
            self.full_df_q_raw = self.full_df_q_raw.sort_values("时间").reset_index(drop=True)
            self.full_df_q_raw["ts"] = mdates.date2num(self.full_df_q_raw["时间"].dt.to_pydatetime())

        self.full_df_v = self.full_df_v_raw.copy()
        self.full_df_q = self.full_df_q_raw.copy()

        if compress_zero:
            if not self.full_df_v.empty:
                self.full_df_v = self.compress_zero_segments(self.full_df_v, "流速v")
            if not self.full_df_q.empty:
                self.full_df_q = self.compress_zero_segments(self.full_df_q, "质量流量")

        if not self.full_df_v.empty:
            self.full_df_v = self.full_df_v.sort_values("时间").reset_index(drop=True)
            self.full_df_v["ts"] = mdates.date2num(self.full_df_v["时间"].dt.to_pydatetime())

        if not self.full_df_q.empty:
            self.full_df_q = self.full_df_q.sort_values("时间").reset_index(drop=True)
            self.full_df_q["ts"] = mdates.date2num(self.full_df_q["时间"].dt.to_pydatetime())

        self._plot_cache_v.clear()
        self._plot_cache_q.clear()
        self._data_version += 1

        self.current_xmin = None
        self.current_xmax = None

        self._ensure_events_connected()
        self.render_current_view()

    def get_plot_df(self, df, y_col, x_min=None, x_max=None, cache=None, kind="v"):
        if df is None or df.empty:
            return pd.DataFrame(columns=["时间", "ts", y_col])

        target_points = self.get_target_points()
        cache_key = self._make_cache_key(x_min, x_max, target_points, kind)

        if cache is not None and cache_key in cache:
            return cache[cache_key]

        view_df = df
        if x_min is not None and x_max is not None:
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            view_df = df[(df["ts"] >= x_min) & (df["ts"] <= x_max)]

        if view_df.empty:
            result = view_df
        else:
            result = self.downsample_minmax(view_df, "ts", y_col, target_points=target_points)

        if cache is not None:
            cache[cache_key] = result
            if len(cache) > 30:
                first_key = next(iter(cache))
                cache.pop(first_key, None)

        return result

    def render_current_view(self, x_min=None, x_max=None):
        self.current_xmin = x_min
        self.current_xmax = x_max

        self.ax1.clear()
        self.ax2.clear()

        df_v_plot = self.get_plot_df(
            self.full_df_v,
            y_col="流速v",
            x_min=x_min,
            x_max=x_max,
            cache=self._plot_cache_v,
            kind="v"
        )

        df_q_plot = self.get_plot_df(
            self.full_df_q,
            y_col="质量流量",
            x_min=x_min,
            x_max=x_max,
            cache=self._plot_cache_q,
            kind="q"
        )

        if not df_v_plot.empty:
            self.ax1.plot(
                df_v_plot["时间"],
                df_v_plot["流速v"].to_numpy(),
                linewidth=self.line_width
            )

        if not df_q_plot.empty:
            self.ax2.plot(
                df_q_plot["时间"],
                df_q_plot["质量流量"].to_numpy(),
                linewidth=self.line_width
            )

        self.ax1.set_title(self.title1, fontsize=self.title_fontsize, fontweight="bold")
        self.ax1.set_ylabel("流速v", fontsize=self.label_fontsize, fontweight="bold")
        self.ax1.grid(True)
        self.ax1.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_ts))

        self.ax2.set_title(self.title2, fontsize=self.title_fontsize, fontweight="bold")
        self.ax2.set_xlabel("时间", fontsize=self.label_fontsize, fontweight="bold")
        self.ax2.set_ylabel("质量流量", fontsize=self.label_fontsize, fontweight="bold")
        self.ax2.grid(True)
        self.ax2.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_ts))

        # 隐藏上图 x 轴标签，避免与下图标题重叠
        self.ax1.tick_params(axis="x", which="both", labelbottom=False)

        for ax in (self.ax1, self.ax2):
            for label in ax.get_yticklabels():
                label.set_fontsize(self.tick_fontsize)
                label.set_fontweight("bold")

        for label in self.ax2.get_xticklabels():
            label.set_fontsize(self.tick_fontsize)
            label.set_fontweight("bold")
            label.set_rotation(30)
            label.set_horizontalalignment("right")

        self.figure.subplots_adjust(
            left=0.07,
            right=0.98,
            top=0.95,
            bottom=0.10,
            hspace=0.20
        )

        self.canvas.draw_idle()

    def on_select_xrange(self, xmin, xmax):
        if xmin is None or xmax is None:
            return
        if abs(xmax - xmin) < 1e-12:
            return

        if xmin > xmax:
            xmin, xmax = xmax, xmin

        self.render_current_view(xmin, xmax)

    def reset_full_view(self):
        self.render_current_view(None, None)

    def open_day_selector(self):
        if not self.date_list:
            QMessageBox.warning(self, "提示", "没有可选日期")
            return

        dlg = DateSelectDialog(self.date_list, self)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.raise_()
        dlg.activateWindow()

        if dlg.exec_() != QDialog.Accepted:
            return

        selected_date = dlg.selected_date()
        if not selected_date:
            return

        df_v_day = pd.DataFrame()
        df_q_day = pd.DataFrame()

        if self.df_v_all is not None and not self.df_v_all.empty:
            df_v_day = self.df_v_all[
                self.df_v_all["时间"].dt.strftime("%Y-%m-%d") == selected_date
            ].copy()

        if self.df_q_all is not None and not self.df_q_all.empty:
            df_q_day = self.df_q_all[
                self.df_q_all["时间"].dt.strftime("%Y-%m-%d") == selected_date
            ].copy()

        if df_v_day.empty and df_q_day.empty:
            QMessageBox.warning(self, "提示", f"{selected_date} 没有数据")
            return

        day_dialog = HistoryPlotDialog(
            parent=self.parent(),
            title=f"{selected_date} 历史流速/质量流量曲线",
            allow_day_select=False
        )
        day_dialog.plot_history(
            df_v_day,
            df_q_day,
            compress_zero=False,
            thin_all_history=False,
            title1=f"{selected_date} 历史流速曲线",
            title2=f"{selected_date} 历史质量流量曲线"
        )
        day_dialog.showFullScreen()
        day_dialog.raise_()
        day_dialog.activateWindow()
        day_dialog.exec_()
class Simplified_MainForm(Simplified_MainWindow):
    triggerFlow = pyqtSignal()
    startFlowLoopSig = pyqtSignal(str, int)
    stopFlowLoopSig = pyqtSignal()
    startCrossLoopSig = pyqtSignal(int)
    stopCrossLoopSig = pyqtSignal()
    startBottomfaceSig = pyqtSignal()
    def __init__(self, Simplified_MainWindow):
        super().__init__()
        # 存储历史流速、截面积
        self.history_len = 400
        self.v_history = deque(maxlen=self.history_len)
        self.area_history = deque(maxlen=self.history_len)
        self.Q_prev = 0.0
        self.prev_gray = None
        self.ref_precomp = None
        self.stop_get = False
        super().setupUi(Simplified_MainWindow)
        #self.apply_simple_ui_style()
        self.Height_choose = 0
        self.Width_choose = 0
        self.area = 0.0
        self.flowrate_device_manager = None
        self.flowrate_cam = None  # 相机
        self.flowrate_cam_state = False  # 相机是否打开
        self.crosssect_device_manager = None
        self.crosssect_cam = None
        self.crosssect_cam_state = False
        self.uimode = 0  # 用户界面
        self.black_mean_threshold = 18  # ROI平均灰度低于该值，认为近乎全黑
        self.black_ratio_threshold = 0.95  # 暗像素占比超过该值，认为近乎全黑
        self.black_pixel_threshold = 25
        # xFeat 流速缓冲与异常检测参数
        self.xfeat_speed_buffer = deque(maxlen=5)  # 最近稳定速度
        self.xfeat_max_jump_ratio = 0.5  # 相对均值最大跳变 50%
        self.xfeat_abs_max_speed = 2.5  # 物理最大速度
        self.xfeat_min_valid_speed = 0.0  # 最小有效速度
        self.xfeat_min_points = 60  # 最少有效匹配点
        self.xfeat_min_dt = 1e-3  # 最小时间差，防止除零/极小dt
        self.xfeat_max_dt = 4  # 最大时间差，卡顿太久直接丢弃
        self.xfeat_last_valid_v = 0.0  # 最近一次有效速度
        self.enable_flow_prediction = True
        self.predict_every_n_frames = 80
        self.predict_history_len = 60
        self.predict_horizon = 5
        self.last_mass_rate_flush = time.time()
        self.predicted_v = 0.0
        self.prediction_buffer = []  # 保存到 CSV
        self.prediction_table_rows = []  # 表格可用
        self.last_prediction_flush = time.time()

        config = self.read_camera_config()
        if config is not None:
            self.enable_flow_prediction = bool(int(config.get("enable_flow_prediction", 0)))
            self.predict_every_n_frames = int(config.get("predict_every_n_frames", 80))
            self.predict_history_len = int(config.get("predict_history_len", 60))
            self.predict_horizon = int(config.get("predict_horizon", 5))
        # ===== 低频缓存写盘 =====
        self.flowrate_buffer = []
        self.crosssect_buffer = []
        self.traffic_buffer = []
        self.mass_rate_buffer = []
        self.current_minute_str = None
        self.current_minute_flow_samples = []  # 当前分钟内的实时流量样本，单位 m³/s
        self.current_minute_volume = 0.0  # 当前分钟流量，单位 m³/min（严格说是当前分钟累计体积）
        self.last_sample_ts = None
        self.simple_preview_scale = 0.5
        self.total_mass = 0.0  # t
        self.last_mass_update_ts = None
        #self.label_minute_flow_value.setText("0.000")
        self.label_v_value.setText("0.000")
        self.label_q_value.setText("0.000")
        self.label_mass_rate_value.setText("0.000")
        self.label_total_mass_value.setText("0.000")
        # ===== 视频保存相关 =====
        self.is_saving_video = False
        self.video_writer = None
        self.video_save_fps = 15  # 可按需调整，10~20 都可以
        self.video_scale = 0.5  # 和可视化类似，降低分辨率保存
        self.current_video_path = None
        self.img_flowrate_camera = None  # 最近一次用于简化界面显示/保存的帧（BGR）
        self.last_video_write_time = 0.0  # 控制写盘帧率
        self.last_flowrate_flush = time.time()
        self.last_crosssect_flush = time.time()
        self.last_traffic_flush = time.time()

        self.flush_interval = 5.0  # 每 5 秒落盘一次
        title_font = QFont("Microsoft YaHei", 20, QFont.Bold)
        self.latest_cross_bgr = None  # 最近一帧截面积图像
        self.zero_flow_start_time = None  # 零流速开始时间
        self.zero_flow_duration = 60  # 连续零流速阈值（秒）
        self.zero_flow_threshold = 0.02  # 判定“接近0流速”的阈值，别直接用 == 0
        self.bottom_contour_updated = False  # 本轮零流速期间是否已经自动更新过

        # 加载flowrate_Qchart波形界面
        self.flowrate_plot_qchart = flowrate_QChartViewPlot()
        self.flowrate_plot_qchart.setTitle("流速")
        self.flowrate_plot_qchart.setTitleFont(title_font)
        self.flowrate_plot_qchart.setMargins(QMargins(20, 20, 20, 20))
        self.simple_flowrate_plot_view.setChart(self.flowrate_plot_qchart)
        self.simple_flowrate_plot_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.simple_flowrate_plot_view.setRubberBand(QChartView.RectangleRubberBand)

        # 加载crosssect_Qchart波形界面
        self.crosssect_plot_qchart = crosssect_QChartViewPlot()
        self.crosssect_plot_qchart.setTitle("截面积")
        self.crosssect_plot_qchart.setTitleFont(title_font)

        self.crosssect_plot_qchart.setMargins(QMargins(20, 20, 20, 20))
        self.simple_crosssect_plot_view.setChart(self.crosssect_plot_qchart)
        self.simple_crosssect_plot_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.simple_crosssect_plot_view.setRubberBand(QChartView.RectangleRubberBand)

        # 加载traffic_Qchart波形界面
        self.traffic_plot_qchart = traffic_QChartViewPlot()
        self.traffic_plot_qchart.setTitle("流量")
        self.traffic_plot_qchart.setTitleFont(title_font)
        self.traffic_plot_qchart.setMargins(QMargins(20, 20, 20, 20))
        self.simple_traffic_plot_view.setChart(self.traffic_plot_qchart)
        self.simple_traffic_plot_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.simple_traffic_plot_view.setRubberBand(QChartView.RectangleRubberBand)
        # 加载预测Qchart波形界面
        self.flowrate_predict_plot_qchart = flowrate_predict_QChartViewPlot()
        self.flowrate_predict_plot_qchart.setTitle("流速预测")
        self.flowrate_predict_plot_qchart.setTitleFont(title_font)
        self.flowrate_predict_plot_qchart.setMargins(QMargins(20, 20, 20, 20))
        self.simple_predict_plot_view.setChart(self.flowrate_predict_plot_qchart)
        self.simple_predict_plot_view.setRenderHint(QPainter.Antialiasing)
        self.simple_predict_plot_view.setRubberBand(QChartView.RectangleRubberBand)

        # 按键状态初始化
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        self.openprojectui.setEnabled(True)

        self.trafficThread = QThread(self)
        self.trafficWorker = TrafficWorker(self)
        self.trafficWorker.moveToThread(self.trafficThread)
        self.trafficThread.start()

        self.BottomfaceThread = QThread(self)
        self.BottomfaceWorker = BottomfaceWorker(self)
        self.BottomfaceWorker.moveToThread(self.BottomfaceThread)
        self.BottomfaceThread.start()

        self.CrossLoopThread = QThread(self)
        self.CrossLoopWorker = CrossLoopWorker(self)
        self.CrossLoopWorker.moveToThread(self.CrossLoopThread)
        self.CrossLoopThread.start()

        self.CrossThread = QThread(self)
        self.CrossWorker = CrossCaptureWorker(self)
        self.CrossWorker.moveToThread(self.CrossThread)
        self.CrossThread.start()

        self.CrossCaculateThread = QThread(self)
        self.CrossCaculateWorker = CrossCaculateWorker(self)
        self.CrossCaculateWorker.moveToThread(self.CrossCaculateThread)
        self.CrossCaculateThread.start()

        self.CrossViewerThread = QThread(self)
        self.CrossViewerWorker = CrossViewerWorker(self)
        self.CrossViewerWorker.moveToThread(self.CrossViewerThread)
        self.CrossViewerThread.start()

        self.flowThread = QThread(self)
        self.flowWorker = FlowrateCaptureWorker(self)
        self.flowWorker.moveToThread(self.flowThread)
        self.flowThread.start()

        self.flowCaculateThread = QThread(self)
        self.flowCaculateWorker = FlowrateCaculateWorker(self)
        self.flowCaculateWorker.moveToThread(self.flowCaculateThread)
        self.flowCaculateThread.start()

        self.flowViewerThread = QThread(self)
        self.flowViewerWorker = FlowrateViewerWorker(self)
        self.flowViewerWorker.moveToThread(self.flowViewerThread)
        self.flowViewerThread.start()

        self.flowLoopThread = QThread(self)
        self.flowLoopWorker = FlowrateLoopWorker(self)
        self.flowLoopWorker.moveToThread(self.flowLoopThread)
        self.flowLoopThread.start()

        self.trafficLoopThread = QThread(self)
        self.trafficLoopWorker = TrafficLoopWorker(self)
        self.trafficLoopWorker.moveToThread(self.trafficLoopThread)
        self.trafficLoopThread.start()
        # 槽信号连接
        #self.start.clicked.connect(lambda: self.measurement("Trad"))
        self.start.clicked.connect(lambda: self.measurement("xFeat"))
        self.start.clicked.connect(self.crosssect_measurement)
        self.stop.clicked.connect(self.flowrate_CloseCamera)
        self.stop.clicked.connect(self.crosssect_CloseCamera)
        self.openprojectui.clicked.connect(self.open_login_window)
        self.history_plot_btn.clicked.connect(self.show_history_plot)
        self.start.clicked.connect(self.start_traffic_loop)
        self.return_btn.clicked.connect(self.exit_program)
        self.button_simple_savevideo.clicked.connect(self.toggle_save_video)
        #流速保存
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = None
        # 流量计时器占位
        self.traffic_start_ts = 0
        self.df_traffic = pd.DataFrame(columns=["时间", "质量流量"])
        self.traffic = []  # 用于存储截面积数据
        self.traffic_times = []  # 用于存储时间戳
        self.traffic_frame_idx = 0  # 初始化帧索引
        self.traffic_iters = 0
        # 截面积计时器占位
        self._start_ts = 0
        self.df_crosssect = pd.DataFrame(columns=["时间", "截面积"])
        self.crosssect_areas = []  # 用于存储截面积数据
        self.crosssect_times = []  # 用于存储时间戳
        self.crosssect_frame_idx = 0  # 初始化帧索引
        self.crosssect_iters = 0

        self._history_cache_v = None
        self._history_cache_q = None
        self._history_cache_mtime = None
        # 测速的一些参数
        self._frame_start_time = 0
        self._frame_end_time = 0
        date = datetime.now()
        timestamp = date.timestamp() * 100
        self.curr_time = timestamp
        self.prev_time = timestamp
        self.e2 = cv2.getTickCount()
        self.e1 = cv2.getTickCount()
        self.df = pd.DataFrame(columns=["时间", "流速v"])
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
        self.flowcamframerate_get = 0  # 流速采集帧率
        self.crosscamframerate_get = 0  # 截面积采集帧率
        self.iters = 1 # 每多少帧输出一次速度
        self.angle = 0 # 筛选方向角度
        self.frame_counter = 0

        self.min_inliers = 50
        self.ransac_thr = 4.0
        self.H = None


        self.duration = 30  # 保存时间

        self.transform = 0.00036237026692619034  # 每个像素代表的长度（m）

        # 图表绘制所需参数
        self.speeds = []  # 添加一个属性来存储每一帧的速度值
        self.frames = []  # 添加一个属性来存储对应的帧数
        self.operate_time = []  # 单帧处理时间
        self.fps = []  # 当前帧率
        self.detected = []  # 检测点数

    def exit_program(self):
        QApplication.instance().quit()
    def open_login_window(self):
        # 打开登录窗口
        login_window = LoginWindow()  # 创建登录窗口
        if login_window.exec_() == QDialog.Accepted:  # 如果密码正确
            self.open_main_window()
    def open_main_window(self):
        # 打开主界面 MainForm
        print("登录成功，打开主界面")
        if self.flowrate_cam_state == True:
            self.flowrate_CloseCamera()
        if self.crosssect_cam_state == True:
            self.crosssect_CloseCamera()

        top = self.centralwidget.window()
        #top.hide()

        def _launch():
            #top.close()
            self.main_window = QMainWindow()
            self.main_ui = MainForm(self.main_window)
            self.main_window.show()

        QTimer.singleShot(0, _launch)

    def toggle_save_video(self):
        """开始/结束保存流速相机视频"""
        if not self.is_saving_video:
            self.start_save_video()
        else:
            self.stop_save_video()

    def start_save_video(self):
        """开始保存简化界面中的流速视频（保存降分辨率后的可视化帧）"""
        try:
            os.makedirs("result", exist_ok=True)

            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.current_video_path = os.path.join("result", f"flowrate_camera_{now_str}.avi")

            # 尝试用最近一帧初始化尺寸；如果还没有帧，就按一个默认尺寸先建
            if self.img_flowrate_camera is not None and self.img_flowrate_camera.size > 0:
                frame = self.img_flowrate_camera
                h, w = frame.shape[:2]
            else:
                # 兜底：还没拿到首帧时，先根据 QLabel 大小给个默认分辨率
                w = max(320, self.simple_flowrate_camera.width())
                h = max(240, self.simple_flowrate_camera.height())

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                self.current_video_path,
                fourcc,
                self.video_save_fps,
                (w, h)
            )

            if self.video_writer is None or not self.video_writer.isOpened():
                self.video_writer = None
                QMessageBox.critical(self, "错误", "视频写入器创建失败。")
                return

            self.is_saving_video = True
            self.last_video_write_time = 0.0
            self.button_simple_savevideo.setText("结束保存视频")

            print(f"开始保存视频: {self.current_video_path}")

        except Exception as e:
            self.video_writer = None
            self.is_saving_video = False
            QMessageBox.critical(self, "错误", f"开始保存视频失败：{e}")

    def stop_save_video(self):
        """结束保存视频"""
        try:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            saved_path = self.current_video_path
            self.current_video_path = None
            self.is_saving_video = False
            self.button_simple_savevideo.setText("开始保存视频")

            if saved_path:
                QMessageBox.information(self, "提示", f"视频已保存到：\n{saved_path}")

            print(f"结束保存视频: {saved_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"结束保存视频失败：{e}")

    def write_video_frame(self, frame):
        """写入一帧视频，frame 必须是已经降分辨率后的 BGR 图"""
        try:
            if not self.is_saving_video or self.video_writer is None:
                return
            if frame is None or frame.size == 0:
                return

            now_ts = time.time()
            interval = 1.0 / max(1, self.video_save_fps)

            # 按设定帧率节流，避免实际写入频率过高
            if self.last_video_write_time > 0 and (now_ts - self.last_video_write_time) < interval:
                return

            self.last_video_write_time = now_ts

            save_frame = frame.copy()

            # 灰度图转 BGR
            if len(save_frame.shape) == 2:
                save_frame = cv2.cvtColor(save_frame, cv2.COLOR_GRAY2BGR)

            # 确保尺寸和 writer 初始化尺寸一致
            # 第一次写入后，不再动态改变 writer 尺寸
            self.video_writer.write(save_frame)

        except Exception as e:
            print(f"写入视频帧失败: {e}")
    def crosssect_measurement(self):
        config = self.read_camera_config()


        # 初始化设备管理器
        self.crosssect_device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self.crosssect_device_manager.update_device_list()
        if dev_num == 0:
            QMessageBox.critical(self, "Error", "未找到截面积相机设备")
            return
        str_sn = config.get("crosssect_cam_sn", None)
        print(f"crosssect_cam_sn：{str_sn}")

        try:
            self.crosssect_cam = self.crosssect_device_manager.open_device_by_sn(str_sn)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"打开截面积相机失败: {e}")
            self.crosssect_cam_state = False
            return
        # 加载相机
        self.crosssect_cam_state = True
        target_fps = config.get("crosssect_cam_target_fps", None)
        if target_fps is not None:
            self.try_set_camera_fps(self.crosssect_cam, target_fps, "crosssect_cam")
        # 读取并输出当前帧率
        self.crosscamframerate_get = self.crosssect_cam.CurrentAcquisitionFrameRate.get()
        self.crosssect_iters = int(self.crosscamframerate_get)
        print("crosscamframerate_get",self.crosscamframerate_get)
        self.crosssect_plot_qchart.timer.start()
        # 白平衡
        self.crosssect_cam.BalanceWhiteAuto.set(gx.GxAutoEntry.OFF)
        for channel, ratio in [("GREEN", int(config.get("crosssect_cam_green", None))), ("RED", int(config.get("crosssect_cam_red", None))), ("BLUE", int(config.get("crosssect_cam_blue", None)))]:
            self.crosssect_cam.BalanceRatioSelector.set(getattr(gx.GxBalanceRatioSelectorEntry, channel))
            self.crosssect_cam.BalanceRatio.set(ratio)

        # 曝光
        #self.crosssect_cam.ExposureTime.set(int(config.get("crosssect_cam_ExposureTime", None)))
        self.crosssect_cam.ExposureTime.set(400)
        # 触发设置（软触发 + 延迟 + 上升沿）
        self.crosssect_cam.TriggerMode.set(gx.GxSwitchEntry.ON)
        self.crosssect_cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        self.crosssect_cam.TriggerDelay.set(int(config.get("TRIGGER_DELAY_US", None)))
        self.crosssect_cam.TriggerActivation.set(gx.GxTriggerActivationEntry.RISINGEDGE)

        # Line1 设置为 STROBE 输出（可选）
        self.crosssect_cam.LineSelector.set(gx.GxLineSelectorEntry.LINE1)
        self.crosssect_cam.LineMode.set(gx.GxLineModeEntry.OUTPUT)
        self.crosssect_cam.LineSource.set(gx.GxLineSourceEntry.STROBE)

        self.BottomfaceThread.start()
        self.CrossLoopThread.start()
        self.CrossThread.start()
        self.CrossCaculateThread.start()
        self.CrossViewerThread.start()
        try:
            self.crosssect_cam.stream_on()
        except Exception as e:
            print(f"crosssect_cam.stream_on() 失败: {e}")
        interval_ms = 3000
        self.CrossLoopWorker.start(interval_ms)

    def start_traffic_loop(self):
        # 流量单独按 1 秒计算一次；你也可以改成 500ms
        interval_ms = 150
        self.trafficLoopWorker.start(interval_ms)

    def try_set_camera_fps(self, cam, target_fps, cam_name="camera"):
        """尝试设置相机帧率。不同机型节点可用性不同，失败时只打印日志，不中断。"""
        try:
            if cam is None:
                return

            target_fps = float(target_fps)
            if target_fps <= 0:
                return

            # 某些机型需要先打开帧率控制开关
            try:
                cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
                print(f"{cam_name}: AcquisitionFrameRateMode 已开启")
            except Exception as e:
                print(f"{cam_name}: 无 AcquisitionFrameRateMode 或开启失败: {e}")

            # 尝试直接设置目标帧率
            try:
                cam.AcquisitionFrameRate.set(target_fps)
                print(f"{cam_name}: 目标帧率已设置为 {target_fps}")
            except Exception as e:
                print(f"{cam_name}: 设置 AcquisitionFrameRate 失败: {e}")

            # 读取当前实际帧率
            try:
                real_fps = cam.CurrentAcquisitionFrameRate.get()
                print(f"{cam_name}: 当前实际帧率 = {real_fps}")
            except Exception as e:
                print(f"{cam_name}: 读取 CurrentAcquisitionFrameRate 失败: {e}")

        except Exception as e:
            print(f"{cam_name}: 帧率设置异常: {e}")

    def flowrate_CloseCamera(self):
        if self.is_saving_video:
            self.stop_save_video()
        # 关闭相机
        self.flowrate_cam_state = False
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        if self.flowrate_cam is not None:
            self.flowrate_cam.stream_off()
            self.flowrate_cam.close_device()
        self.flowrate_plot_qchart.timer.stop()
        self.traffic_plot_qchart.timer.stop()
        self.flowrate_predict_plot_qchart.timer.stop()

        self.flowrate_plot_qchart.series.clear()
        self.traffic_plot_qchart.series.clear()
        self.flowrate_predict_plot_qchart.series_real.clear()
        self.flowrate_predict_plot_qchart.series_pred.clear()
        # # 清空三条曲线，不销毁对象
        # self.arima_plot_qchart.orig_series.clear()
        # self.arima_plot_qchart.pred_series.clear()
        # self.arima_plot_qchart.filt_series.clear()

        self.flowThread.quit()
        self.flowThread.wait()
        self.flowCaculateThread.quit()
        self.flowCaculateThread.wait()
        self.flowLoopThread.quit()
        self.flowLoopThread.wait()
        self.trafficLoopThread.quit()
        self.trafficLoopThread.wait()
        self.flowViewerThread.quit()
        self.flowViewerThread.wait()
        self.trafficThread.quit()
        self.trafficThread.wait()
        self.flowWorker.reset()
        self.flowLoopWorker.reset()
        self.flowViewerWorker.reset()
        # 关闭前把缓存写完
        self.append_rows_to_csv(
            "result/realtime_flowrate.csv",
            self.flowrate_buffer,
            ["时间", "流速v"]
        )
        self.flowrate_buffer.clear()

        self.append_rows_to_csv(
            "result/realtime_traffic.csv",
            self.traffic_buffer,
            ["时间", "流量"]
        )

        self.append_rows_to_csv(
            "result/total_mass.csv",
            self.total_mass,
            ["时间", "总质量"]
        )
        self.save_total_mass_file()
        self.append_rows_to_csv(
            "result/realtime_flow_prediction.csv",
            self.prediction_buffer,
            ["时间", "当前流速v", "预测流速v"]
        )
        self.prediction_buffer.clear()
        self.traffic_buffer.clear()


    def crosssect_CloseCamera(self):
        self.crosssect_plot_qchart.timer.stop()  # 关闭曲线显示

        self.crosssect_plot_qchart.series.clear()

        self.CrossThread.quit()
        self.CrossThread.wait()
        self.CrossCaculateThread.quit()
        self.CrossCaculateThread.wait()
        self.CrossLoopThread.quit()
        self.CrossLoopThread.wait()
        self.CrossViewerThread.quit()
        self.CrossViewerThread.wait()
        self.BottomfaceThread.quit()
        self.BottomfaceThread.wait()
        self.CrossViewerWorker.reset()
        self.CrossCaculateWorker.reset()
        self.CrossLoopWorker.reset()



        self.append_rows_to_csv(
            "result/realtime_crosssect_area.csv",
            self.crosssect_buffer,
            ["时间", "截面积"]
        )
        self.crosssect_buffer.clear()
        # 关闭相机
        if self.crosssect_cam is not None:
            self.crosssect_cam.close_device()
            self.crosssect_cam.stream_off()
        self.crosssect_cam_state = False


    def append_rows_to_csv(self, file_path, rows, columns):
        """
        将 rows（list[dict]）追加写入 CSV。
        不在实时循环中写 Excel，只做低频 CSV 追加。
        """
        if not rows:
            return
        try:
            df = pd.DataFrame(rows, columns=columns)
            file_exists = os.path.exists(file_path)
            df.to_csv(
                file_path,
                mode='a',
                header=not file_exists,
                index=False,
                encoding='utf-8-sig'
            )
        except Exception as e:
            print(f"追加写入 CSV 失败 {file_path}: {e}")



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

    def read_camera_config(self):
        config_file = resource_path('ParamConfig.txt')
        config_dict = {}
        try:
            with open(config_file, 'r') as file:
                config_lines = file.readlines()
                for line in config_lines:
                    # 跳过空行或注释行
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # 按 '=' 分割每一行，获取参数名和参数值
                    key, value = line.split("=", 1)
                    config_dict[key.strip()] = value.strip()

            return config_dict  # 返回包含所有配置的字典
        except Exception as e:
            QMessageBox.critical(self, "Error", f"读取配置文件失败: {str(e)}")
            return None

    def check_auto_update_bottom_contour(self):
        """
        简化界面下：
        当流速连续接近 0 达到 zero_flow_duration 秒后，
        自动使用当前截面积图像更新下表面轮廓。
        """
        try:
            # 相机必须在线
            if not self.crosssect_cam_state:
                self.zero_flow_start_time = None
                self.bottom_contour_updated = False
                return

            # 用“接近0”而不是“等于0”
            if abs(self.v_t) <= self.zero_flow_threshold:
                if self.zero_flow_start_time is None:
                    self.zero_flow_start_time = time.time()
                else:
                    elapsed = time.time() - self.zero_flow_start_time
                    if elapsed >= self.zero_flow_duration and not self.bottom_contour_updated:
                        self.auto_update_bottom_contour()
                        print("自动更新下底面轮廓,",time.time())
                        self.bottom_contour_updated = True
            else:
                # 只要流速恢复，就重置计时和触发标志
                self.zero_flow_start_time = None
                self.bottom_contour_updated = False

        except Exception as e:
            print(f"检查自动更新下表面轮廓失败: {e}")

    def auto_update_bottom_contour(self):
        try:
            if self.latest_cross_bgr is None:
                print("自动更新下表面轮廓失败：没有可用的截面积图像")
                return

            image = self.latest_cross_bgr.copy()

            # 固定 ROI 参数，建议从配置文件读取
            config = self.read_camera_config()
            x1 = int(config.get("bottom_roi_x1", 100))
            y1 = int(config.get("bottom_roi_y1", 200))
            x2 = int(config.get("bottom_roi_x2", 600))
            y2 = int(config.get("bottom_roi_y2", 500))


            w = x2 - x1
            h = y2 - y1
            roi_rect = (x1, y1, w, h)

            if roi_rect is None or roi_rect.size == 0:
                print("自动更新下表面轮廓失败：ROI 无效")
                return

            save_lower_countour(image, SAVE_ROI=False, roi_rect=roi_rect)

            print("已自动更新下表面轮廓（固定ROI）")

        except Exception as e:
            print(f"自动更新下表面轮廓失败: {e}")

    def apply_simple_ui_style(self):
        self.setStyleSheet("""
            QWidget {
                font-size: 24px;
            }
            QPushButton {
                font-size: 28px;
                min-height: 72px;
                border-radius: 10px;
                padding: 8px 16px;
            }
            QLineEdit, QDoubleSpinBox {
                font-size: 26px;
                min-height: 56px;
                padding: 6px 10px;
            }
            QLabel {
                font-size: 26px;
            }
            QListWidget {
                font-size: 24px;
            }
        """)
    def measurement(self, method="Trad"):

        config = self.read_camera_config()

        #加载相机
        self.flowrate_cam_state = True
        # 初始化设备管理器
        self.flowrate_device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self.flowrate_device_manager.update_device_list()
        if dev_num == 0:
            QMessageBox.critical(self, "Error", "未找到流速相机设备")
            return
        str_sn = config.get("flowrate_cam_sn", None)
        print(f"flowrate_cam_sn：{str_sn}")
        # 打开选中设备
        try:
            self.flowrate_cam = self.flowrate_device_manager.open_device_by_sn(str_sn)
            print(f"相机已成功打开: SN={str_sn}")
        except Exception as e:
            self.flowrate_cam = None
            print(f"打开相机失败: SN={str_sn}, 错误信息: {e}")
        target_fps = config.get("flowrate_cam_target_fps", None)
        if target_fps is not None:
            self.try_set_camera_fps(self.flowrate_cam, target_fps, "flowrate_cam")
        # 读取并输出当前帧率
        self.flowcamframerate_get = self.flowrate_cam.CurrentAcquisitionFrameRate.get()
        #self.iters = int(self.flowcamframerate_get)
        print("flowcamframerate_get", self.flowcamframerate_get)
        # 自动曝光 / 自动增益 / 自动白平衡
        self.flowrate_cam.ExposureMode.set(int(config.get("flowrate_cam_ExposureMode", None)))
        self.flowrate_cam.AutoExposureTimeMax.set(int(config.get("flowrate_cam_AutoExposureTimeMax", None)))
        self.flowrate_cam.GainAuto.set(int(config.get("flowrate_cam_GainAuto", None)))
        self.flowrate_cam.BalanceWhiteAuto.set(int(config.get("flowrate_cam_BalanceWhiteAuto", None)))
        # self.flowrate_cam.BalanceRatio.set(100)    # 白平衡参数设置
        self.flowrate_cam.ExposureTime.set(int(config.get("flowrate_cam_ExposureTime", None)))  # 曝光时间设置
        # self.flowrate_cam.Gain.set(10.0)           # 增益参数设置
        # TRIGGER_DELAY_US = int(config.get("TRIGGER_DELAY_US", 7))
        #
        # self.flowrate_cam.TriggerMode.set(gx.GxSwitchEntry.ON)
        # self.flowrate_cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        # self.flowrate_cam.TriggerDelay.set(TRIGGER_DELAY_US)
        # self.flowrate_cam.TriggerActivation.set(gx.GxTriggerActivationEntry.RISINGEDGE)

        self.flowrate_cam.LineSelector.set(gx.GxLineSelectorEntry.LINE1)
        self.flowrate_cam.LineMode.set(gx.GxLineModeEntry.OUTPUT)
        self.flowrate_cam.LineSource.set(gx.GxLineSourceEntry.USER_OUTPUT0)  # 用一个用户输出作为源
        self.flowrate_cam.UserOutputValue.set(False)  # 设为低电平

        # 切换按钮状态
        self.start.setEnabled(False)
        self.stop.setEnabled(True)

        self.flowThread.start()
        self.flowCaculateThread.start()
        self.flowViewerThread.start()
        self.flowLoopThread.start()
        self.trafficLoopThread.start()
        try:
            self.flowrate_cam.stream_on()
        except Exception as e:
            print(f"flowrate_cam.stream_on() 失败: {e}")
        # 保存视频
        self.out = cv2.VideoWriter("save_video/" + now_str_ms() + ".mp4",
                                   self.fourcc,
                                   24, (self.flowrate_cam.Width.get(), self.flowrate_cam.Height.get()))

        self.flowrate_plot_qchart.timer.start()  # 开始曲线
        self.traffic_plot_qchart.timer.start()
        self.flowrate_predict_plot_qchart.timer.start()
        self.e1 = cv2.getTickCount()

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
        interval_ms = int(1000 / self.flowcamframerate_get)

        self.flowWorker.image_ready.connect(lambda bgr_image: self.flowCaculateWorker.serial_prev_gray(bgr_image))
        self.flowWorker.start_capture.connect(self.flowWorker.bgrimage_capture)
        self.flowCaculateWorker.done_prev_gray.connect(
            lambda: self.flowLoopWorker.start(
                method,
                interval_ms
            ),
            Qt.QueuedConnection
        )

        self.flowWorker.start_capture.emit()



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

    def save_total_mass_file(self):
        try:
            import os
            from datetime import datetime

            os.makedirs("result", exist_ok=True)

            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_path = os.path.join("result", f"总质量_{time_str}.txt")

            # 这里改成你程序里真实保存“总质量”的变量名
            total_mass = getattr(self, "total_mass", 0.0)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"总质量: {total_mass:.6f} kg\n")
                f.write(f"保存时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")



        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存总质量文件失败：{e}")

    def update_simple_camera_view(self, bgr_image):
        """在简化界面显示当前流速采集画面，并可选保存显示用的降分辨率视频"""
        try:
            if bgr_image is None or bgr_image.size == 0:
                return

            scale = max(0.1, min(float(self.simple_preview_scale), 1.0))
            if scale < 0.999:
                small = cv2.resize(
                    bgr_image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA
                )
            else:
                small = bgr_image.copy()

            # 保存最近一帧“简化界面显示帧”，供 start_save_video 初始化和后续写盘使用
            self.img_flowrate_camera = small.copy()

            # 正在录像时，把当前显示帧写入文件
            if self.is_saving_video:
                self.write_video_frame(self.img_flowrate_camera)

            # 显示到 QLabel
            pixmap = self.CvMatToQImage(small)
            if isinstance(pixmap, QPixmap):
                label_w = self.simple_flowrate_camera.width()
                label_h = self.simple_flowrate_camera.height()
                pixmap = pixmap.scaled(
                    label_w,
                    label_h,
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation
                )
                self.simple_flowrate_camera.setPixmap(pixmap)

        except Exception as e:
            print(f"更新简化界面视频失败: {e}")

    def update_simple_values(self):
        try:
            self.label_v_value.setText(f"{self.v_t:.3f}")

            if len(self.traffic) > 0:
                current_q = self.traffic[-1]
            else:
                current_q = self.v_t * self.area

            self.label_q_value.setText(f"{current_q:.3f}")
            self.update_flow_and_mass(current_q)

            # 低频预测：不要每帧都跑
            if self.enable_flow_prediction and self.frame_counter % self.predict_every_n_frames == 0:
                self.predict_flow_speed_lightweight()

        except Exception as e:
            print(f"更新简化界面数值失败: {e}")

    # def predict_flow_speed_lightweight(self):
    #     """轻量流速预测：基于最近历史做低频预测，避免每帧卡顿"""
    #     try:
    #         if len(self.v_history) < max(10, self.predict_history_len // 3):
    #             return
    #
    #         data = list(self.v_history)[-self.predict_history_len:]
    #         arr = np.array(data, dtype=np.float32)
    #
    #         # 方法1：均值 + 最近趋势
    #         recent_mean = float(np.mean(arr[-10:])) if len(arr) >= 10 else float(np.mean(arr))
    #         if len(arr) >= 6:
    #             trend = float(np.mean(np.diff(arr[-6:])))
    #         else:
    #             trend = 0.0
    #
    #         pred = recent_mean + trend * self.predict_horizon
    #
    #         # 简单限幅，避免离谱
    #         pred = max(0.0, min(pred, 5.0))
    #         self.predicted_v = pred
    #
    #         # ===== 刷新预测数值 =====
    #         if hasattr(self, "label_pred_v_value"):
    #             self.label_pred_v_value.setText(f"{self.predicted_v:.3f}")
    #
    #         # ===== 刷新预测曲线 =====
    #         if hasattr(self, "flowrate_predict_plot_qchart"):
    #             self.flowrate_predict_plot_qchart.update_value(self.v_t, self.predicted_v)
    #
    #         now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         row = {
    #             "时间": now_str,
    #             "当前流速v": round(self.v_t, 4),
    #             "预测流速v": round(self.predicted_v, 4)
    #         }
    #         self.prediction_buffer.append(row)
    #         self.prediction_table_rows.append(row)
    #
    #         # ===== 刷新预测表格 =====
    #         if hasattr(self, "prediction_table"):
    #             self.update_prediction_table(row)
    #
    #         now_ts = time.time()
    #         if now_ts - self.last_prediction_flush >= self.flush_interval:
    #             self.append_rows_to_csv(
    #                 "result/realtime_flow_prediction.csv",
    #                 self.prediction_buffer,
    #                 ["时间", "当前流速v", "预测流速v"]
    #             )
    #             self.prediction_buffer.clear()
    #             self.last_prediction_flush = now_ts
    #
    #     except Exception as e:
    #         print(f"低频流速预测失败: {e}")
    def predict_flow_speed_lightweight(self):
        """低频流速预测：直接在本函数内使用 ARIMA"""
        try:
            # 历史数据太少时不预测
            min_need = max(20, self.predict_history_len // 2)
            if len(self.v_history) < min_need:
                return

            data = list(self.v_history)[-self.predict_history_len:]
            arr = np.array(data, dtype=np.float64)

            # 去掉异常值，避免 ARIMA 拟合发散
            #arr = np.clip(arr, 0.0, 5.0)

            # 全常数序列时没必要拟合
            if np.allclose(arr, arr[0]):
                pred = float(arr[-1])
            else:
                # ARIMA 拟合
                # 这里先固定用 (1,1,1)，对大多数平稳/缓变序列比较稳
                model = ARIMA(arr, order=(1, 1, 1))
                fitted = model.fit()

                # 预测未来 self.predict_horizon 个点，取最后一个
                forecast = fitted.forecast(steps=max(1, int(self.predict_horizon)))
                pred = float(forecast[-1])

            # 简单限幅，避免离谱
           # pred = max(0.0, min(pred, 5.0))
            self.predicted_v = pred

            # ===== 刷新预测数值 =====
            if hasattr(self, "label_pred_v_value"):
                self.label_pred_v_value.setText(f"{self.predicted_v:.3f}")

            # ===== 刷新预测曲线 =====
            if hasattr(self, "flowrate_predict_plot_qchart"):
                self.flowrate_predict_plot_qchart.update_value(self.v_t, self.predicted_v)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = {
                "时间": now_str,
                "当前流速v": round(self.v_t, 4),
                "预测流速v": round(self.predicted_v, 4)
            }
            self.prediction_buffer.append(row)
            self.prediction_table_rows.append(row)

            # ===== 刷新预测表格 =====
            if hasattr(self, "prediction_table"):
                self.update_prediction_table(row)

            now_ts = time.time()
            if now_ts - self.last_prediction_flush >= self.flush_interval:
                self.append_rows_to_csv(
                    "result/realtime_flow_prediction.csv",
                    self.prediction_buffer,
                    ["时间", "当前流速v", "预测流速v"]
                )
                self.prediction_buffer.clear()
                self.last_prediction_flush = now_ts

        except Exception as e:
            print(f"低频流速预测失败: {e}")
    def update_prediction_table(self, row):
        """更新预测表格，只保留最近50条"""
        try:
            if not hasattr(self, "prediction_table"):
                return

            table = self.prediction_table
            max_rows = 50

            current_rows = table.rowCount()
            table.insertRow(current_rows)

            table.setItem(current_rows, 0, QTableWidgetItem(str(row["时间"])))
            table.setItem(current_rows, 1, QTableWidgetItem(str(row["当前流速v"])))
            table.setItem(current_rows, 2, QTableWidgetItem(str(row["预测流速v"])))

            # 只保留最近 max_rows 条
            while table.rowCount() > max_rows:
                table.removeRow(0)

            table.scrollToBottom()
        except Exception as e:
            print(f"更新预测表格失败: {e}")
    def update_flow_and_mass(self, current_q):
        """
        current_q: 当前实时流量，单位 m³/s

        统一完成：
        1. 当前分钟平均流量（m³/s）
        2. 当前分钟流量（m³/min）
        3. 当前质量流量（t/min）
        4. 当前总质量（t，积分）
        """
        try:
            now_dt = datetime.now()
            now_ts = time.time()
            minute_str = now_dt.strftime("%Y-%m-%d %H:%M")

            density = self.label_density_value.value()  # t/m³

            # ---------- 初始化 ----------
            if self.current_minute_str is None:
                self.current_minute_str = minute_str
                self.current_minute_flow_samples = []
                self.current_minute_volume = 0.0
                self.last_sample_ts = now_ts
                self.last_mass_update_ts = now_ts

                #self.label_minute_flow_value.setText("0.000")
                self.label_mass_rate_value.setText("0.000")
                self.label_total_mass_value.setText(f"{self.total_mass:.3f}")
                return

            # ---------- 计算 dt ----------
            if self.last_sample_ts is None:
                dt = 0.0
            else:
                dt = now_ts - self.last_sample_ts
            self.last_sample_ts = now_ts

            # 防止异常 dt
            if dt < 0:
                dt = 0.0
            if dt > 2.0:
                # 界面卡顿、线程切换等可能导致 dt 突然很大，做个上限保护
                dt = 2.0

            # ---------- 总质量积分 ----------
            # M += Q * rho * dt
            if dt > 0:
                self.total_mass += current_q * density * dt  # t

            # ---------- 分钟切换 ----------
            if minute_str != self.current_minute_str:
                self.current_minute_str = minute_str
                self.current_minute_flow_samples = []
                self.current_minute_volume = 0.0

            # ---------- 本分钟累计体积 ----------
            # V_minute += Q * dt
            if dt > 0:
                self.current_minute_volume += current_q * dt  # m³

            self.current_minute_flow_samples.append(current_q)

            # ---------- 当前分钟已过时间 ----------
            elapsed_sec = now_dt.second + now_dt.microsecond / 1e6
            if elapsed_sec <= 0:
                elapsed_sec = 1e-6

            # 当前分钟平均流量（m³/s）
            avg_q_this_min = self.current_minute_volume / elapsed_sec

            # 当前分钟流量（m³/min）
            minute_flow_m3_min = avg_q_this_min * 60.0

            # 当前质量流量（t/min）
            mass_rate_t_min = minute_flow_m3_min * density

            # ---------- 界面显示 ----------
            #self.label_minute_flow_value.setText(f"{minute_flow_m3_min:.3f}")
            self.label_mass_rate_value.setText(f"{mass_rate_t_min:.3f}")
            self.label_total_mass_value.setText(f"{self.total_mass:.3f}")
            # ---------- 保存实时质量流量历史 ----------
            now_str = now_str_ms()
            row = {
                "时间": now_str,
                "质量流量": round(mass_rate_t_min, 4)
            }
            self.mass_rate_buffer.append(row)

            if now_ts - self.last_mass_rate_flush >= self.flush_interval:
                self.append_rows_to_csv(
                    "result/realtime_mass_rate.csv",
                    self.mass_rate_buffer,
                    ["时间", "质量流量"]
                )
                self.mass_rate_buffer.clear()
                self.last_mass_rate_flush = now_ts

        except Exception as e:
            print(f"更新流量/质量失败: {e}")


    def show_history_plot(self):
        flowrate_file = os.path.join("result", "realtime_flowrate.csv")
        mass_rate_file = os.path.join("result", "realtime_mass_rate.csv")

        try:
            if not os.path.exists(flowrate_file) or not os.path.exists(mass_rate_file):
                QMessageBox.warning(self, "提示", "历史文件不存在")
                return

            mtime = (
                os.path.getmtime(flowrate_file),
                os.path.getmtime(mass_rate_file)
            )

            if self._history_cache_mtime != mtime:
                df_v = pd.read_csv(flowrate_file, encoding="utf-8-sig", usecols=["时间", "流速v"])
                df_q = pd.read_csv(mass_rate_file, encoding="utf-8-sig", usecols=["时间", "质量流量"])

                df_v.columns = df_v.columns.astype(str).str.strip()
                df_q.columns = df_q.columns.astype(str).str.strip()

                def parse_time_column(series):
                    s = series.astype(str).str.strip()

                    # 先尝试新格式：2026-04-22 14:35:12.123
                    dt = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

                    # 对旧格式兜底：2026-04-22 14-35-12
                    mask = dt.isna()
                    if mask.any():
                        dt.loc[mask] = pd.to_datetime(
                            s.loc[mask],
                            format="%Y-%m-%d %H-%M-%S",
                            errors="coerce"
                        )

                    # 再兜底一次，防止混杂格式
                    mask = dt.isna()
                    if mask.any():
                        dt.loc[mask] = pd.to_datetime(s.loc[mask], errors="coerce")

                    return dt
                df_v["时间"] = parse_time_column(df_v["时间"])
                df_q["时间"] = parse_time_column(df_q["时间"])

                df_v["流速v"] = pd.to_numeric(df_v["流速v"], errors="coerce")
                df_q["质量流量"] = pd.to_numeric(df_q["质量流量"], errors="coerce")

                df_v = df_v.dropna(subset=["时间", "流速v"]).sort_values("时间").reset_index(drop=True)
                df_q = df_q.dropna(subset=["时间", "质量流量"]).sort_values("时间").reset_index(drop=True)

                self._history_cache_v = df_v
                self._history_cache_q = df_q
                self._history_cache_mtime = mtime
            else:
                df_v = self._history_cache_v.copy()
                df_q = self._history_cache_q.copy()

            date_set = set()
            if not df_v.empty:
                date_set.update(df_v["时间"].dt.strftime("%Y-%m-%d").dropna().unique())
            if not df_q.empty:
                date_set.update(df_q["时间"].dt.strftime("%Y-%m-%d").dropna().unique())

            date_list = sorted(list(date_set), reverse=True)

            self.history_dialog = HistoryPlotDialog(
                parent=self,
                title="全部历史流速/质量流量曲线",
                allow_day_select=True,
                date_list=date_list
            )
            self.history_dialog.plot_history(
                df_v,
                df_q,
                compress_zero=False,
                thin_all_history=False,
                title1="全部历史流速曲线",
                title2="全部历史质量流量曲线"
            )
            self.history_dialog.showFullScreen()
            self.history_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取历史曲线失败：{e}")
class MainForm(Ui_mainWindow):
    triggerFlow = pyqtSignal()
    startFlowLoopSig = pyqtSignal(str, int)
    stopFlowLoopSig = pyqtSignal()
    startCrossLoopSig = pyqtSignal(int)
    stopCrossLoopSig = pyqtSignal()
    startBottomfaceSig = pyqtSignal()
    def __init__(self, MainForm):
        super().__init__()
        # 存储历史流速、截面积
        self.history_len = 10
        self.v_history = deque(maxlen=self.history_len)
        self.area_history = deque(maxlen=self.history_len)
        self.vis_view = None
        self.Q_prev = 0.0
        self.prev_gray = None
        self.area = 0.0
        self.ref_precomp = None
        self.stop_get = False
        super().setupUi(MainForm)
        self.Height_choose = 0
        self.Width_choose = 0
        self.flowrate_device_manager = None
        self.flowrate_cam = None  # 相机
        self.flowrate_cam_state = False  # 相机是否打开
        self.crosssect_device_manager = None
        self.crosssect_cam = None
        self.crosssect_cam_state = False
        self.uimode = 1#配置界面
        self.black_pixel_threshold = 25
        self.black_mean_threshold = 18  # ROI平均灰度低于该值，认为近乎全黑
        self.black_ratio_threshold = 0.95  # 暗像素占比超过该值，认为近乎全黑
        # ===== 低频缓存写盘 =====
        self.flowrate_buffer = []
        self.crosssect_buffer = []
        self.traffic_buffer = []
        self.mass_rate_buffer = []
        # xFeat 流速缓冲与异常检测参数
        self.xfeat_speed_buffer = deque(maxlen=5)  # 最近稳定速度
        self.xfeat_max_jump_ratio = 0.5  # 相对均值最大跳变 50%
        self.xfeat_abs_max_speed = 4.0  # 物理最大速度
        self.xfeat_min_valid_speed = 0.0  # 最小有效速度
        self.xfeat_min_points = 60  # 最少有效匹配点
        self.xfeat_min_dt = 1e-3  # 最小时间差，防止除零/极小dt
        self.xfeat_max_dt = 0.5  # 最大时间差，卡顿太久直接丢弃
        self.xfeat_last_valid_v = 0.0  # 最近一次有效速度
        self.last_flowrate_flush = time.time()
        self.last_crosssect_flush = time.time()
        self.last_traffic_flush = time.time()

        self.flush_interval = 5.0  # 每 5 秒落盘一次
        # 加载flowrate_Qchart波形界面
        self.flowrate_plot_qchart = flowrate_QChartViewPlot()
        self.flowrate_plot_qchart.setTitle("速度")
        self.flowrate_plot_qchart.setMargins(QMargins(20, 20, 20, 20))
        self.flowrate_plot_view.setChart(self.flowrate_plot_qchart)
        self.flowrate_plot_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.flowrate_plot_view.setRubberBand(QChartView.RectangleRubberBand)

        # 加载crosssect_Qchart波形界面
        self.crosssect_plot_qchart = crosssect_QChartViewPlot()
        self.crosssect_plot_qchart.setTitle("截面积")
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
        self.flowratecamera_param_control.setEnabled(False)
        self.crosssectcamera_param_control.setEnabled(False)
        self.return_ui.setEnabled(True) 

        self.BottomfaceThread = QThread(self)
        self.BottomfaceWorker = BottomfaceWorker(self)
        self.BottomfaceWorker.moveToThread(self.BottomfaceThread)
        self.BottomfaceThread.start()

        self.CrossLoopThread = QThread(self)
        self.CrossLoopWorker = CrossLoopWorker(self)
        self.CrossLoopWorker.moveToThread(self.CrossLoopThread)
        self.CrossLoopThread.start()

        self.CrossThread = QThread(self)
        self.CrossWorker = CrossCaptureWorker(self)
        self.CrossWorker.moveToThread(self.CrossThread)
        self.CrossThread.start()

        self.CrossCaculateThread = QThread(self)
        self.CrossCaculateWorker = CrossCaculateWorker(self)
        self.CrossCaculateWorker.moveToThread(self.CrossCaculateThread)
        self.CrossCaculateThread.start()

        self.CrossViewerThread = QThread(self)
        self.CrossViewerWorker = CrossViewerWorker(self)
        self.CrossViewerWorker.moveToThread(self.CrossViewerThread)
        self.CrossViewerThread.start()

        self.flowThread = QThread(self)
        self.flowWorker = FlowrateCaptureWorker(self)
        self.flowWorker.moveToThread(self.flowThread)
        self.flowThread.start()

        self.flowCaculateThread = QThread(self)
        self.flowCaculateWorker = FlowrateCaculateWorker(self)
        self.flowCaculateWorker.moveToThread(self.flowCaculateThread)
        self.flowCaculateThread.start()

        self.flowViewerThread = QThread(self)
        self.flowViewerWorker = FlowrateViewerWorker(self)
        self.flowViewerWorker.moveToThread(self.flowViewerThread)
        self.flowViewerThread.start()

        self.flowLoopThread = QThread(self)
        self.flowLoopWorker = FlowrateLoopWorker(self)
        self.flowLoopWorker.moveToThread(self.flowLoopThread)
        self.flowLoopThread.start()

        self.trafficThread = QThread(self)
        self.trafficWorker = TrafficWorker(self)
        self.trafficWorker.moveToThread(self.trafficThread)
        self.trafficThread.start()

        # self.startFlowLoopSig.connect(self.flowLoopWorker.start, Qt.QueuedConnection)
        # self.stopFlowLoopSig.connect(self.flowLoopWorker.reset, Qt.QueuedConnection)
        #
        # self.startCrossLoopSig.connect(self.CrossLoopWorker.start(interval_ms=60), Qt.QueuedConnection)
        # self.stopCrossLoopSig.connect(self.CrossLoopWorker.reset, Qt.QueuedConnection)
        #
        # self.startBottomfaceSig.connect(self.BottomfaceWorker.Bottomface_caculate, Qt.QueuedConnection)
        # 槽信号连接
        self.flowrate_OpenCam.clicked.connect(self.flowrate_OpenCamera)
        self.flowrate_single.clicked.connect(self.flowrate_SingleAcq)
        self.flowrate_CloseCam.clicked.connect(self.flowrate_CloseCamera)
        self.flowrate_continuous.clicked.connect(self.flowrate_ContinuousAcq)
        self.flowrate_choose_roi.clicked.connect(self.flowrate_Choose_ROI)
        self.flowratecamera_param_control.clicked.connect(self.flowrate_param_control)
        self.flowrate_Trad_start.clicked.connect(lambda: self.measurement("Trad"))
        self.flowrate_xFeat_start.clicked.connect(lambda: self.measurement("xFeat"))

        self.crosssect_OpenCam.clicked.connect(self.crosssect_OpenCamera)
        self.crosssect_CloseCam.clicked.connect(self.crosssect_CloseCamera)
        self.crosssect_continuous.clicked.connect(self.crosssect_ContinuousAcq)
        self.crosssect_start.clicked.connect(lambda: self.CrossLoopWorker.start(interval_ms=int(60)))
        self.crosssectcamera_param_control.clicked.connect(self.crosssect_param_control)
        self.bottom_face_start.clicked.connect(self.BottomfaceWorker.Bottomface_caculate)

        self.return_ui.clicked.connect(self.return_clicked)

        #流速保存
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = None
        # 流量计时器占位
        self.traffic_start_ts = 0
        self.df_traffic = pd.DataFrame(columns=["时间", "截面积"])
        self.traffic = []  # 用于存储截面积数据
        self.traffic_times = []  # 用于存储时间戳
        self.traffic_frame_idx = 0  # 初始化帧索引
        self.traffic_iters = 0
        # 截面积计时器占位
        self._start_ts = 0
        self.df_crosssect = pd.DataFrame(columns=["时间", "截面积"])
        self.crosssect_areas = []  # 用于存储截面积数据
        self.crosssect_times = []  # 用于存储时间戳
        self.crosssect_frame_idx = 0  # 初始化帧索引
        self.crosssect_iters = 0

        # 测速的一些参数
        self._frame_start_time = 0
        self._frame_end_time = 0
        date = datetime.now()
        timestamp = date.timestamp() * 100
        self.curr_time = timestamp
        self.prev_time = timestamp
        self.e2 = cv2.getTickCount()
        self.e1 = cv2.getTickCount()
        self.df = pd.DataFrame(columns=["时间", "流速v"])
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
        self.last_time = 0.0
        self.dtime = 0.0
        self.cost_time = 0  # 处理时间
        self.f = 0  # 帧数
        self.flowcamframerate_get = 0  # 流速采集帧率
        self.crosscamframerate_get = 0  # 截面积采集帧率
        self.iters = 1  # 每多少帧输出一次速度
        self.angle = 0  # 筛选方向角度
        self.frame_counter = 0

        self.min_inliers = 50
        self.ransac_thr = 4.0
        self.H = None
        self.ref_precomp = None

        self.duration = 30  # 保存时间

        self.transform = 0.00036237026692619034  # 每个像素代表的长度（m）

        # 图表绘制所需参数
        self.speeds = []  # 添加一个属性来存储每一帧的速度值
        self.frames = []  # 添加一个属性来存储对应的帧数
        self.operate_time = []  # 单帧处理时间
        self.fps = []  # 当前帧率
        self.detected = []  # 检测点数

    def return_clicked(self):
        # 关闭相机等资源
        if self.flowrate_cam_state:
            self.flowrate_CloseCamera()
        if self.crosssect_cam_state:
            self.crosssect_CloseCamera()
        # 直接关闭当前的顶层 QMainWindow（MainForm 窗口）
        self.centralwidget.window().close()

    def append_rows_to_csv(self, file_path, rows, columns):
        """
        将 rows（list[dict]）追加写入 CSV。
        不在实时循环中写 Excel，只做低频 CSV 追加。
        """
        if not rows:
            return
        try:
            df = pd.DataFrame(rows, columns=columns)
            file_exists = os.path.exists(file_path)
            df.to_csv(
                file_path,
                mode='a',
                header=not file_exists,
                index=False,
                encoding='utf-8-sig'
            )
        except Exception as e:
            print(f"追加写入 CSV 失败 {file_path}: {e}")
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
        self.flowcamframerate_get = self.flowrate_cam.CurrentAcquisitionFrameRate.get()
        print(self.flowcamframerate_get)
        #self.iters = int(self.flowcamframerate_get)

        # 自动曝光 / 自动增益 / 自动白平衡
        self.flowrate_cam.ExposureMode.set(1)
        self.flowrate_cam.AutoExposureTimeMax.set(1000000.0)
        self.flowrate_cam.GainAuto.set(1)
        self.flowrate_cam.BalanceWhiteAuto.set(1)
        # self.flowrate_cam.BalanceRatio.set(100)    # 白平衡参数设置
        self.flowrate_cam.ExposureTime.set(50.0)   # 曝光时间设置
        # self.flowrate_cam.Gain.set(10.0)           # 增益参数设置

        TRIGGER_DELAY_US = 7

        # 触发设置（软触发 + 延迟 + 上升沿）
        self.flowrate_cam.TriggerMode.set(gx.GxSwitchEntry.ON)
        self.flowrate_cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        self.flowrate_cam.TriggerDelay.set(TRIGGER_DELAY_US)
        self.flowrate_cam.TriggerActivation.set(gx.GxTriggerActivationEntry.RISINGEDGE)

        self.flowrate_cam.LineSelector.set(gx.GxLineSelectorEntry.LINE1)
        self.flowrate_cam.LineMode.set(gx.GxLineModeEntry.OUTPUT)
        self.flowrate_cam.LineSource.set(gx.GxLineSourceEntry.USER_OUTPUT0)  # 用一个用户输出作为源
        self.flowrate_cam.UserOutputValue.set(False)  # 设为低电平

        # 切换按钮状态
        self.flowrate_OpenCam.setEnabled(False)
        self.flowrate_CloseCam.setEnabled(True)
        self.flowratecamera_param_control.setEnabled(True)


        self.flowThread.start()
        self.flowCaculateThread.start()
        self.flowViewerThread.start()
        self.flowLoopThread.start()


    def flowrate_param_control(self):
        self.flowratecamera_param_control.setEnabled(False)
        """弹出相机参数设置对话框"""
        # 确保相机已打开
        self.flowrate_cam.stream_on()
        # if not hasattr(self.main, 'flowrate_cam') or self.main.flowrate_cam is None:
        #     QMessageBox.warning(self, "提示", "请先打开相机再设置参数")
        #     return

        # 创建对话框
        dialog = QDialog(self,
                         Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
        dialog.setWindowTitle("相机参数编辑")

        layout = QVBoxLayout(dialog)

        # -- 曝光时间 --
        label_exp = QLabel("曝光时间 (μs):", dialog)
        self.flow_exp_input = QSpinBox(dialog)
        # 从相机读取当前曝光时间，作默认值
        try:
            current_exp = self.flowrate_cam.ExposureTime.get()
        except Exception:
            current_exp = 10.0
        self.flow_exp_input.setRange(1, 1000000)
        self.flow_exp_input.setValue(int(current_exp))
        layout.addWidget(label_exp)
        layout.addWidget(self.flow_exp_input)

        # -- 自动白平衡 --
        self.flow_awb_checkbox = QCheckBox("自动白平衡", dialog)
        # 默认根据相机当前状态
        try:
            awb_on = bool(self.flowrate_cam.BalanceWhiteAuto.get())
        except Exception:
            awb_on = True
        self.flow_awb_checkbox.setChecked(awb_on)
        layout.addWidget(self.flow_awb_checkbox)

        # -- 应用按钮 --
        btn_apply = QPushButton("应用", dialog)
        btn_apply.clicked.connect(lambda: self.apply_flowratecamera_params(dialog))
        layout.addWidget(btn_apply)

        dialog.setLayout(layout)
        dialog.exec_()
        self.flowratecamera_param_control.setEnabled(True)
        self.flowrate_single.setEnabled(True)
        self.flowrate_continuous.setEnabled(True)
        self.flowrate_choose_roi.setEnabled(True)
        self.flowrate_cam.stream_off()

    def apply_flowratecamera_params(self, dialog):
        """读取对话框输入并设置相机参数"""
        exposure_time = self.flow_exp_input.value()
        auto_wb = self.flow_awb_checkbox.isChecked()

        try:
            # 设置曝光时间
            self.flowrate_cam.ExposureTime.set(float(exposure_time))
            # 设置自动白平衡
            self.flowrate_cam.BalanceWhiteAuto.set(1 if auto_wb else 0)

            # 日志输出
            self.flowInfo.append(f"✔ 曝光时间已设为 {exposure_time} μs")
            self.flowInfo.append(f"✔ 自动白平衡 {'已启用' if auto_wb else '已禁用'}")

            # 关闭对话框
            dialog.accept()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置相机参数失败：{e}")



    def crosssect_OpenCamera(self):
        EXPOSURE_TIME_US = 50  # 曝光时间
        TRIGGER_DELAY_US = 7

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
        self.crosscamframerate_get = self.crosssect_cam.CurrentAcquisitionFrameRate.get()
        self.crosssect_iters = int(self.crosscamframerate_get)


        # 白平衡
        self.crosssect_cam.BalanceWhiteAuto.set(gx.GxAutoEntry.OFF)
        for channel, ratio in [("GREEN", 15), ("RED", 1), ("BLUE", 1)]:
            self.crosssect_cam.BalanceRatioSelector.set(getattr(gx.GxBalanceRatioSelectorEntry, channel))
            self.crosssect_cam.BalanceRatio.set(ratio)

        # 曝光
        self.crosssect_cam.ExposureTime.set(EXPOSURE_TIME_US)

        # 触发设置（软触发 + 延迟 + 上升沿）
        self.crosssect_cam.TriggerMode.set(gx.GxSwitchEntry.ON)
        self.crosssect_cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        self.crosssect_cam.TriggerDelay.set(TRIGGER_DELAY_US)
        self.crosssect_cam.TriggerActivation.set(gx.GxTriggerActivationEntry.RISINGEDGE)

        # Line1 设置为 STROBE 输出（可选）
        self.crosssect_cam.LineSelector.set(gx.GxLineSelectorEntry.LINE1)
        self.crosssect_cam.LineMode.set(gx.GxLineModeEntry.OUTPUT)
        self.crosssect_cam.LineSource.set(gx.GxLineSourceEntry.STROBE)

        # 切换按钮状态
        self.crosssect_OpenCam.setEnabled(False)
        self.crosssectcamera_param_control.setEnabled(True)
        try:
            self.crosssect_cam.stream_on()
        except Exception as e:
            print(f"crosssect_cam.stream_on() 失败: {e}")
        self.BottomfaceThread.start()
        self.CrossLoopThread.start()
        self.CrossThread.start()
        self.CrossCaculateThread.start()
        self.CrossViewerThread.start()

    def crosssect_param_control(self):
        self.crosssectcamera_param_control.setEnabled(False)
        """弹出相机参数设置对话框"""
        # 确保相机已打开
        self.crosssect_cam.stream_on()
        # if not hasattr(self, 'crosssect_cam') or self.crosssect_cam is None:
        #     QMessageBox.warning(self, "提示", "请先打开相机再设置参数")
        #     return

        # 创建对话框
        dialog = QDialog(self,
                         Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
        dialog.setWindowTitle("相机参数编辑")

        layout = QVBoxLayout(dialog)

        # -- 曝光时间 --
        label_exp = QLabel("曝光时间 (μs):", dialog)
        self.cross_exp_input = QSpinBox(dialog)
        # 从相机读取当前曝光时间，作默认值
        try:
            current_exp = self.crosssect_cam.ExposureTime.get()
        except Exception:
            current_exp = 10.0
        self.cross_exp_input.setRange(1, 1000000)
        self.cross_exp_input.setValue(int(current_exp))
        layout.addWidget(label_exp)
        layout.addWidget(self.cross_exp_input)

        # -- 自动白平衡 --
        self.cross_awb_checkbox = QCheckBox("自动白平衡", dialog)
        # 默认根据相机当前状态
        try:
            awb_on = bool(self.crosssect_cam.BalanceWhiteAuto.get())
        except Exception:
            awb_on = True
        self.cross_awb_checkbox.setChecked(awb_on)
        layout.addWidget(self.cross_awb_checkbox)

        # -- 应用按钮 --
        btn_apply = QPushButton("应用", dialog)
        btn_apply.clicked.connect(lambda: self.apply_crosssectcamera_params(dialog))
        layout.addWidget(btn_apply)

        dialog.setLayout(layout)
        dialog.exec_()
        self.crosssectcamera_param_control.setEnabled(True)
        self.crosssect_CloseCam.setEnabled(True)
        self.crosssect_continuous.setEnabled(True)
        self.crosssect_start.setEnabled(True)
        self.bottom_face_start.setEnabled(True)
        self.bottom_face_start.setEnabled(True)
        self.crosssect_cam.stream_off()

    def apply_crosssectcamera_params(self, dialog):
        """读取对话框输入并设置相机参数"""
        exposure_time = self.cross_exp_input.value()
        auto_wb = self.cross_awb_checkbox.isChecked()

        try:
            # 设置曝光时间
            self.crosssect_cam.ExposureTime.set(float(exposure_time))
            # 设置自动白平衡
            self.crosssect_cam.BalanceWhiteAuto.set(1 if auto_wb else 0)

            # 日志输出
            self.crossInfo.append(f"✔ 曝光时间已设为 {exposure_time} μs")
            self.crossInfo.append(f"✔ 自动白平衡 {'已启用' if auto_wb else '已禁用'}")

            # 关闭对话框
            dialog.accept()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置相机参数失败：{e}")





    def flowrate_CloseCamera(self):

        # 关闭相机
        self.flowrate_cam_state = False
        self.flowrate_OpenCam.setEnabled(True)
        self.flowrate_CloseCam.setEnabled(False)
        self.flowrate_single.setEnabled(False)
        self.flowrate_continuous.setEnabled(False)
        self.flowrate_choose_roi.setEnabled(False)
        self.flowrate_Trad_start.setEnabled(False)
        self.flowrate_xFeat_start.setEnabled(False)
        self.flowratecamera_param_control.setEnabled(False)
        if self.flowrate_cam is not None:
            self.flowrate_cam.stream_off()
            self.flowrate_cam.close_device()
        self.flowrate_plot_qchart.timer.stop()  # 关闭曲线显示
        self.flowrate_plot_qchart.series.clear()

        self.flowThread.quit()
        self.flowThread.wait()
        self.flowCaculateThread.quit()
        self.flowCaculateThread.wait()
        self.flowLoopThread.quit()
        self.flowLoopThread.wait()
        self.flowViewerThread.quit()
        self.flowViewerThread.wait()
        self.flowWorker.reset()
        self.flowLoopWorker.reset()
        self.flowViewerWorker.reset()

        # 关闭前把缓存写完
        self.append_rows_to_csv(
            "result/realtime_flowrate.csv",
            self.flowrate_buffer,
            ["时间", "流速v"]
        )
        self.flowrate_buffer.clear()

        self.append_rows_to_csv(
            "result/realtime_traffic.csv",
            self.traffic_buffer,
            ["时间", "流量"]
        )
        self.append_rows_to_csv(
            "result/realtime_mass_rate.csv",
            self.mass_rate_buffer,
            ["时间", "质量流量"]
        )
        self.mass_rate_buffer.clear()
        self.traffic_buffer.clear()
    def crosssect_CloseCamera(self):
        self.CrossThread.quit()
        self.CrossThread.wait()
        self.CrossCaculateThread.quit()
        self.CrossCaculateThread.wait()
        self.CrossLoopThread.quit()
        self.CrossLoopThread.wait()
        self.CrossViewerThread.quit()
        self.CrossViewerThread.wait()
        self.BottomfaceThread.quit()
        self.BottomfaceThread.wait()
        self.CrossViewerWorker.reset()
        self.CrossCaculateWorker.reset()
        self.CrossLoopWorker.reset()
        # 关闭相机
        self.crosssect_plot_qchart.timer.stop()
        self.crosssect_cam_state = False
        self.crosssect_OpenCam.setEnabled(True)
        self.crosssect_CloseCam.setEnabled(False)
        self.crosssect_continuous.setEnabled(False)
        self.crosssect_start.setEnabled(False)
        self.bottom_face_start.setEnabled(False)
        self.crosssectcamera_param_control.setEnabled(False)
        if self.crosssect_cam is not None:
            self.crosssect_cam.stream_off()
            self.crosssect_cam.close_device()
        self.crosssect_plot_qchart.timer.stop()  # 关闭曲线显示
        # # 清空图表数据
        # for series in self.crosssect_plot_qchart.chart.series():
        #     series.clear()
        # # 重置坐标轴范围（可选）
        # self.crosssect_plot_qchart.axisX.setRange(0, 10)
        # self.crosssect_plot_qchart.axisY.setRange(0, 200)

        self.append_rows_to_csv(
            "result/realtime_crosssect_area.csv",
            self.crosssect_buffer,
            ["时间", "截面积"]
        )
        self.crosssect_buffer.clear()

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
        self.flowWorker.pixmap_ready.connect(lambda pixmap: self.flowViewerWorker.flowrate_Camera(pixmap))

        self.flowWorker.start_capture.emit()

    def flowrate_ContinuousAcq(self):
        self.flowratecamera_param_control.setEnabled(False)
        try:
            self.flowWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.flowWorker.pixmap_ready.disconnect()
        except TypeError:
            pass
        self.flowWorker.start_capture.connect(self.flowWorker.process_flowrate)
        self.flowWorker.pixmap_ready.connect(lambda pixmap: self.flowViewerWorker.flowrate_Camera(pixmap))
        self.flowrate_single.setEnabled(False)
        self.flowrate_continuous.setEnabled(False)
        self.flowrate_choose_roi.setEnabled(False)
        self.flowrate_cam_state = True
        while True:
            if not self.flowrate_cam_state:
                break
            self.flowWorker.start_capture.emit()
            cv2.waitKey(int(1000 / self.flowcamframerate_get))



    def crosssect_ContinuousAcq(self):
        self.crosssectcamera_param_control.setEnabled(False)
        try:
            self.CrossWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.CrossWorker.pixmap_ready.disconnect()
        except TypeError:
            pass
        self.CrossWorker.start_capture.connect(self.CrossWorker.process_cross)
        self.CrossWorker.pixmap_ready.connect(lambda pixmap: self.CrossViewerWorker.cross_Camera(pixmap))
        self.crosssect_continuous.setEnabled(False)
        self.crosssect_start.setEnabled(False)
        self.bottom_face_start.setEnabled(False)

        self.crosssect_cam_state = True
        while True:
            if not self.crosssect_cam_state:
                break
            self.CrossWorker.start_capture.emit()
            cv2.waitKey(int(1000 / self.crosscamframerate_get))






    def flowrate_Choose_ROI(self):
        try:
            self.flowWorker.start_capture.disconnect()
        except TypeError:
            pass
        try:
            self.flowWorker.image_ready.disconnect()
        except TypeError:
            pass
        self.flowWorker.start_capture.connect(self.flowWorker.bgrimage_capture)
        self.flowWorker.image_ready.connect(lambda bgr_image:self.flowViewerWorker.serial_flowrate_Choose_ROI(bgr_image))
        self.flowWorker.start_capture.emit()


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
        self.flowratecamera_param_control.setEnabled(False)
        # 保存视频
        self.out = cv2.VideoWriter("save_video/" + now_str_ms() + ".mp4",
                                   self.fourcc,
                                   24, (self.flowrate_cam.Width.get(), self.flowrate_cam.Height.get()))

        self.flowrate_choose_roi.setEnabled(False)
        self.flowrate_single.setEnabled(False)
        self.flowrate_continuous.setEnabled(False)
        if method == "Trad":
            self.flowrate_Trad_start.setEnabled(False)
        if method == "xFeat":
            self.flowrate_xFeat_start.setEnabled(False)
        self.flowrate_plot_qchart.timer.start()  # 开始曲线

        self.ref_precomp = xfeat.detectAndCompute(self.ref_frame, top_k=1024)[0]
        self.e1 = cv2.getTickCount()


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
        if method == "Trad":
            interval_ms = int(1000 / self.flowcamframerate_get)
        if method == "xFeat":
            interval_ms = int(1000 / self.flowcamframerate_get)

        self.flowWorker.image_ready.connect(lambda bgr_image: self.flowCaculateWorker.serial_prev_gray(bgr_image))
        self.flowWorker.start_capture.connect(self.flowWorker.bgrimage_capture)
        self.flowCaculateWorker.done_prev_gray.connect(
            lambda: self.flowLoopWorker.start(
                method,
                interval_ms
            ),
            Qt.QueuedConnection
        )
        self.last_time = time.time()
        self.flowWorker.start_capture.emit()
        try:
            self.flowrate_cam.stream_on()
        except Exception as e:
            print(f"flowrate_cam.stream_on() 失败: {e}")



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
        self.series.setName("截面积/m^2")

        self.axisX = QDateTimeAxis()
        self.axisY = QValueAxis()

        axis_font = self.axisY.labelsFont()
        axis_font.setPointSize(14)
        axis_font.setBold(True)
        self.axisX.setLabelsFont(axis_font)
        self.axisY.setLabelsFont(axis_font)

        legend_font = self.legend().font()
        legend_font.setPointSize(14)
        legend_font.setBold(True)
        self.legend().setFont(legend_font)

        self.axisX.setFormat("HH:mm:ss")
        self.axisX.setTickCount(4)
        now = QDateTime.currentDateTime()
        self.axisX.setRange(now.addSecs(-10), now)

        self.axisY.setRange(0, 0.05)

        self.timer = QTimer()
        self.timer.timeout.connect(self.handleTimeout)
        self.timer.setInterval(1000)

        redPen = QPen(Qt.red)
        redPen.setWidth(3)
        self.series.setPen(redPen)

        self.addSeries(self.series)
        self.addAxis(self.axisX, Qt.AlignBottom)
        self.addAxis(self.axisY, Qt.AlignLeft)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)

        self.y = 0

    def update_value(self, value):
        self.y = value

    def handleTimeout(self):
        now = QDateTime.currentDateTime()
        x = now.toMSecsSinceEpoch()

        self.series.append(x, self.y)

        min_time = now.addSecs(-60).toMSecsSinceEpoch()
        points = self.series.pointsVector()
        while points and points[0].x() < min_time:
            self.series.remove(0)
            points = self.series.pointsVector()

        self.axisX.setRange(now.addSecs(-60), now)
class flowrate_predict_QChartViewPlot(QChart):
    def __init__(self, parent=None):
        super(flowrate_predict_QChartViewPlot, self).__init__()

        # 当前流速曲线
        self.series_real = QSplineSeries()
        self.series_real.setName("当前流速/m/s")

        # 预测流速曲线
        self.series_pred = QSplineSeries()
        self.series_pred.setName("预测流速/m/s")

        self.axisX = QDateTimeAxis()
        self.axisY = QValueAxis()

        axis_font = self.axisY.labelsFont()
        axis_font.setPointSize(14)
        axis_font.setBold(True)
        self.axisX.setLabelsFont(axis_font)
        self.axisY.setLabelsFont(axis_font)

        legend_font = self.legend().font()
        legend_font.setPointSize(14)
        legend_font.setBold(True)
        self.legend().setFont(legend_font)

        self.axisX.setFormat("HH:mm:ss")
        self.axisX.setTickCount(4)
        now = QDateTime.currentDateTime()
        self.axisX.setRange(now.addSecs(-10), now)

        self.axisY.setRange(0, 3)

        realPen = QPen(Qt.red)
        realPen.setWidth(3)
        self.series_real.setPen(realPen)

        predPen = QPen(Qt.blue)
        predPen.setWidth(3)
        self.series_pred.setPen(predPen)

        self.addSeries(self.series_real)
        self.addSeries(self.series_pred)

        self.addAxis(self.axisX, Qt.AlignBottom)
        self.addAxis(self.axisY, Qt.AlignLeft)

        self.series_real.attachAxis(self.axisX)
        self.series_real.attachAxis(self.axisY)
        self.series_pred.attachAxis(self.axisX)
        self.series_pred.attachAxis(self.axisY)

        self.timer = QTimer()
        self.timer.timeout.connect(self.handleTimeout)
        self.timer.setInterval(1000)

        self.y_real = 0
        self.y_pred = 0

    def update_value(self, real_value, pred_value):
        self.y_real = real_value
        self.y_pred = pred_value

    def handleTimeout(self):
        now = QDateTime.currentDateTime()
        x = now.toMSecsSinceEpoch()

        self.series_real.append(x, self.y_real)
        self.series_pred.append(x, self.y_pred)

        # 只保留最近60秒
        min_time = now.addSecs(-60).toMSecsSinceEpoch()

        points_real = self.series_real.pointsVector()
        while points_real and points_real[0].x() < min_time:
            self.series_real.remove(0)
            points_real = self.series_real.pointsVector()

        points_pred = self.series_pred.pointsVector()
        while points_pred and points_pred[0].x() < min_time:
            self.series_pred.remove(0)
            points_pred = self.series_pred.pointsVector()

        self.axisX.setRange(now.addSecs(-60), now)

        # 自适应Y轴
        ys = [p.y() for p in self.series_real.pointsVector()] + [p.y() for p in self.series_pred.pointsVector()]
        if len(ys) > 0:
            ymax = max(ys)
            ymin = min(ys)
            low = max(0, ymin - 0.2)
            high = max(1.0, ymax + 0.2)
            self.axisY.setRange(low, high)
class traffic_QChartViewPlot(QChart):
    def __init__(self, parent=None):
        super(traffic_QChartViewPlot, self).__init__()
        self.series = QSplineSeries()
        self.series.setName("流量/m^3/s")

        self.axisX = QDateTimeAxis()
        self.axisY = QValueAxis()

        axis_font = self.axisY.labelsFont()
        axis_font.setPointSize(14)
        axis_font.setBold(True)
        self.axisX.setLabelsFont(axis_font)
        self.axisY.setLabelsFont(axis_font)

        legend_font = self.legend().font()
        legend_font.setPointSize(14)
        legend_font.setBold(True)
        self.legend().setFont(legend_font)

        self.axisX.setFormat("HH:mm:ss")
        self.axisX.setTickCount(4)
        now = QDateTime.currentDateTime()
        self.axisX.setRange(now.addSecs(-10), now)

        self.axisY.setRange(0, 0.15)

        self.timer = QTimer()
        self.timer.timeout.connect(self.handleTimeout)
        self.timer.setInterval(1000)

        redPen = QPen(Qt.red)
        redPen.setWidth(3)
        self.series.setPen(redPen)

        self.addSeries(self.series)
        self.addAxis(self.axisX, Qt.AlignBottom)
        self.addAxis(self.axisY, Qt.AlignLeft)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)

        self.y = 0

    def update_value(self, value):
        self.y = value

    def handleTimeout(self):
        now = QDateTime.currentDateTime()
        x = now.toMSecsSinceEpoch()

        self.series.append(x, self.y)

        min_time = now.addSecs(-60).toMSecsSinceEpoch()
        points = self.series.pointsVector()
        while points and points[0].x() < min_time:
            self.series.remove(0)
            points = self.series.pointsVector()

        self.axisX.setRange(now.addSecs(-60), now)
class flowrate_QChartViewPlot(QChart):
    def __init__(self, parent=None):
        super(flowrate_QChartViewPlot, self).__init__()
        self.series = QSplineSeries()
        self.series.setName("速度/m/s")

        # 改成时间轴
        self.axisX = QDateTimeAxis()
        self.axisY = QValueAxis()

        # 坐标轴字体
        axis_font = self.axisY.labelsFont()
        axis_font.setPointSize(14)
        axis_font.setBold(True)
        self.axisX.setLabelsFont(axis_font)
        self.axisY.setLabelsFont(axis_font)

        # 图例字体
        legend_font = self.legend().font()
        legend_font.setPointSize(14)
        legend_font.setBold(True)
        self.legend().setFont(legend_font)

        # 时间轴显示格式
        self.axisX.setFormat("HH:mm:ss")
        self.axisX.setTickCount(4)
        now = QDateTime.currentDateTime()
        self.axisX.setRange(now.addSecs(-10), now)

        self.axisY.setRange(0, 3)

        redPen = QPen(Qt.red)
        redPen.setWidth(3)
        self.series.setPen(redPen)

        self.addSeries(self.series)
        self.addAxis(self.axisX, Qt.AlignBottom)
        self.addAxis(self.axisY, Qt.AlignLeft)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)

        self.timer = QTimer()
        self.timer.timeout.connect(self.handleTimeout)
        self.timer.setInterval(1000)

        self.y = 0

    def update_value(self, value):
        self.y = value

    def handleTimeout(self):
        now = QDateTime.currentDateTime()
        x = now.toMSecsSinceEpoch()

        self.series.append(x, self.y)

        # 只保留最近 60 秒
        min_time = now.addSecs(-60).toMSecsSinceEpoch()
        points = self.series.pointsVector()
        while points and points[0].x() < min_time:
            self.series.remove(0)
            points = self.series.pointsVector()

        self.axisX.setRange(now.addSecs(-60), now)
# class crosssect_QChartViewPlot(QChart):
#     def __init__(self, parent=None):
#         super(crosssect_QChartViewPlot, self).__init__()
#         self.series = QSplineSeries()
#         self.series.setName("截面积/m^2")
#         self.axisX = QValueAxis()
#         self.axisY = QValueAxis()
#         self.step = 500
#         self.x = 0
#         self.y = 0
#
#         # ===== 坐标轴字体 =====
#         axis_font = self.axisX.labelsFont()
#         axis_font.setPointSize(14)
#         axis_font.setBold(True)
#         self.axisX.setLabelsFont(axis_font)
#         self.axisY.setLabelsFont(axis_font)
#
#         # ===== 图例字体 =====
#         legend_font = self.legend().font()
#         legend_font.setPointSize(14)
#         legend_font.setBold(True)
#         self.legend().setFont(legend_font)
#
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.handleTimeout)
#         self.timer.setInterval(1000)
#
#         redPen = QPen(Qt.red)
#         redPen.setWidth(3)
#         self.series.setPen(redPen)
#         self.series.append(self.x, self.y)
#
#         self.addSeries(self.series)
#
#         self.addAxis(self.axisX, Qt.AlignBottom)
#         self.addAxis(self.axisY, Qt.AlignLeft)
#         self.series.attachAxis(self.axisX)
#         self.series.attachAxis(self.axisY)
#         self.axisX.setTickCount(4)
#         self.axisX.setRange(-10, 0)
#         self.axisY.setRange(0, 0.2)
#
#     def handleTimeout(self):
#         x = self.plotArea().width() / self.axisX.tickCount()
#         y_step = (self.axisX.max() - self.axisX.min()) / self.axisX.tickCount()
#         self.x += y_step
#         self.series.append(self.x, self.y)
#         self.scroll(x, 0)
#
#
# class traffic_QChartViewPlot(QChart):
#     def __init__(self, parent=None):
#         super(traffic_QChartViewPlot, self).__init__()
#         self.series = QSplineSeries()
#         self.series.setName("流量/m^3/s")
#         self.axisX = QValueAxis()
#         self.axisY = QValueAxis()
#         self._start_ts = time.time()
#         self.step = 500
#         self.x = 0
#         self.y = 0
#
#         # ===== 坐标轴字体 =====
#         axis_font = self.axisX.labelsFont()
#         axis_font.setPointSize(14)
#         axis_font.setBold(True)
#         self.axisX.setLabelsFont(axis_font)
#         self.axisY.setLabelsFont(axis_font)
#
#         # ===== 图例字体 =====
#         legend_font = self.legend().font()
#         legend_font.setPointSize(14)
#         legend_font.setBold(True)
#         self.legend().setFont(legend_font)
#
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.handleTimeout)
#         self.timer.setInterval(1000)
#
#         redPen = QPen(Qt.red)
#         redPen.setWidth(3)
#         self.series.setPen(redPen)
#         self.series.append(self.x, self.y)
#
#         self.addSeries(self.series)
#
#         self.addAxis(self.axisX, Qt.AlignBottom)
#         self.addAxis(self.axisY, Qt.AlignLeft)
#         self.series.attachAxis(self.axisX)
#         self.series.attachAxis(self.axisY)
#         self.axisX.setTickCount(4)
#         self.axisX.setRange(-10, 0)
#         self.axisY.setRange(0, 0.3)
#
#     def handleTimeout(self):
#         x = self.plotArea().width() / self.axisX.tickCount()
#         y_step = (self.axisX.max() - self.axisX.min()) / self.axisX.tickCount()
#         self.x += y_step
#         self.series.append(self.x, self.y)
#         self.scroll(x, 0)
#
#
# class flowrate_QChartViewPlot(QChart):
#     def __init__(self, parent=None):
#         super(flowrate_QChartViewPlot, self).__init__()
#         self.series = QSplineSeries()
#         self.series.setName("速度/m/s")
#         self.axisX = QValueAxis()
#         self.axisY = QValueAxis()
#         self.step = 500
#         self.x = 0
#         self.y = 0
#
#         # ===== 坐标轴字体 =====
#         axis_font = self.axisX.labelsFont()
#         axis_font.setPointSize(14)
#         axis_font.setBold(True)
#         self.axisX.setLabelsFont(axis_font)
#         self.axisY.setLabelsFont(axis_font)
#
#         # ===== 图例字体 =====
#         legend_font = self.legend().font()
#         legend_font.setPointSize(14)
#         legend_font.setBold(True)
#         self.legend().setFont(legend_font)
#
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.handleTimeout)
#         self.timer.setInterval(1000)
#
#         redPen = QPen(Qt.red)
#         redPen.setWidth(3)
#         self.series.setPen(redPen)
#         self.series.append(self.x, self.y)
#
#         self.addSeries(self.series)
#
#         self.addAxis(self.axisX, Qt.AlignBottom)
#         self.addAxis(self.axisY, Qt.AlignLeft)
#         self.series.attachAxis(self.axisX)
#         self.series.attachAxis(self.axisY)
#         self.axisX.setTickCount(4)
#         self.axisX.setRange(-10, 0)
#         self.axisY.setRange(0, 4)
#
#         # self.timer.start()
#
#     def handleTimeout(self):
#         x = self.plotArea().width() / self.axisX.tickCount()
#         y = (self.axisX.max() - self.axisX.min()) / self.axisX.tickCount()
#         self.x += y
#         self.series.append(self.x, self.y)
#         self.scroll(x, 0)

class arima_QChartViewPlot(QChart):
    def __init__(self, parent=None):
        super(arima_QChartViewPlot, self).__init__(parent)

        # 坐标轴
        self.axisX = QValueAxis()
        self.axisY = QValueAxis()

        font = self.axisX.labelsFont()
        font.setPointSize(7)
        self.axisX.setLabelsFont(font)

        font = self.axisY.labelsFont()
        font.setPointSize(7)
        self.axisY.setLabelsFont(font)

        self.addAxis(self.axisX, Qt.AlignBottom)
        self.addAxis(self.axisY, Qt.AlignLeft)

        self.axisX.setTickCount(4)
        self.axisX.setRange(0, 10)
        self.axisY.setRange(0, 10)

        # ===== 固定三条曲线（不用每次 add/remove） =====
        self.orig_series = QSplineSeries()
        self.orig_series.setName("原始数据")
        self.orig_series.setColor(Qt.blue)
        self.addSeries(self.orig_series)
        self.orig_series.attachAxis(self.axisX)
        self.orig_series.attachAxis(self.axisY)

        self.pred_series = QSplineSeries()
        self.pred_series.setName("ARIMA预测")
        self.pred_series.setColor(Qt.red)
        self.addSeries(self.pred_series)
        self.pred_series.attachAxis(self.axisX)
        self.pred_series.attachAxis(self.axisY)

        self.filt_series = QSplineSeries()
        self.filt_series.setName("卡尔曼滤波")
        self.filt_series.setColor(Qt.green)
        self.addSeries(self.filt_series)
        self.filt_series.attachAxis(self.axisX)
        self.filt_series.attachAxis(self.axisY)

        # 永久开启图例
        self.legend().setVisible(True)
        self.legend().setAlignment(Qt.AlignTop)

    def update_chart(self, sub_data, predict_data, result_data):

        # 转 list，兼容 numpy/pandas/deque
        sub_data = list(sub_data) if sub_data is not None else []
        predict_data = list(predict_data) if predict_data is not None else []
        result_data = list(result_data) if result_data is not None else []

        # 清空数据后追加
        self.orig_series.clear()
        for i, v in enumerate(sub_data):
            self.orig_series.append(i, v)

        self.pred_series.clear()
        for i, v in enumerate(predict_data):
            self.pred_series.append(i, v)

        self.filt_series.clear()
        for i, v in enumerate(result_data):
            self.filt_series.append(i, v)

        # 自动调整坐标范围
        all_data = sub_data + predict_data + result_data
        if all_data:
            min_y = min(all_data)
            max_y = max(all_data)
            if min_y == max_y:  # 防止平直
                min_y -= 1
                max_y += 1
            self.axisX.setRange(0, max(len(sub_data), len(predict_data), len(result_data)))
            self.axisY.setRange(min_y * 0.9, max_y * 1.1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    with open(resource_path('design.qss'), 'r', encoding='utf-8') as f:
        app.setStyleSheet(f.read())
    splash = SplashWithLog()
    splash.show()
    app.processEvents()
    do_initialization()
    MainWindow = QMainWindow()  # 主窗口
    Simplified_MainForm(MainWindow)
    splash.close()
    MainWindow.showFullScreen()
    sys.exit(app.exec_())
