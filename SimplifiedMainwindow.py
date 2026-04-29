# -*- coding: utf-8 -*-

from datetime import datetime

from PyQt5.QtGui import QFont, QPainter
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtChart import QChartView
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        self.setGeometry(500, 300, 200, 100)

        self.layout = QVBoxLayout()

        self.password_label = QLabel("请输入密码")
        self.layout.addWidget(self.password_label)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.password_input)

        self.login_button = QPushButton("登录")
        self.layout.addWidget(self.login_button)

        self.setLayout(self.layout)
        self.login_button.clicked.connect(self.check_password)

    def check_password(self):
        password = self.password_input.text()
        if password == "1":
            self.accept()
        else:
            QMessageBox.warning(self, "错误", "密码错误，请重试！")


class Simplified_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 900)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        self.root_layout = QVBoxLayout(self.centralwidget)
        self.root_layout.setContentsMargins(12, 12, 12, 12)
        self.root_layout.setSpacing(12)

        # ===================== 顶部区 =====================
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        self.start = QtWidgets.QPushButton("开始流量检测")
        self.stop = QtWidgets.QPushButton("结束流量检测")
        self.history_plot_btn = QtWidgets.QPushButton("历史曲线")
        self.openprojectui = QtWidgets.QPushButton("设置")
        self.button_simple_savevideo = QtWidgets.QPushButton("开始保存视频")
        self.return_btn = QtWidgets.QPushButton("返回")
        btn_font = QFont("Microsoft YaHei", 28)
        btn_font.setBold(True)

        for btn in [self.start, self.stop,self.return_btn, self.history_plot_btn, self.openprojectui,self.button_simple_savevideo]:
            btn.setFont(btn_font)
            btn.setMinimumHeight(80)

        left_btn_layout = QHBoxLayout()
        left_btn_layout.setSpacing(12)
        left_btn_layout.addWidget(self.start)
        left_btn_layout.addWidget(self.stop)
        left_btn_layout.addWidget(self.return_btn)
        center_layout = QVBoxLayout()
        center_layout.setSpacing(4)

        self.label_title = QLabel("熔渣流速-截面积综合在线测量系统")
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName("label_title")

        self.label_current_time = QLabel("0000-00-00 00:00:00")
        self.label_current_time.setAlignment(QtCore.Qt.AlignCenter)
        time_font = QFont("Microsoft YaHei", 18)
        time_font.setBold(True)
        self.label_current_time.setFont(time_font)

        center_layout.addWidget(self.label_title)
        center_layout.addWidget(self.label_current_time)

        right_btn_layout = QHBoxLayout()
        right_btn_layout.setSpacing(12)
        right_btn_layout.addWidget(self.history_plot_btn)
        right_btn_layout.addWidget(self.openprojectui)
        right_btn_layout.addWidget(self.button_simple_savevideo)
        top_bar.addLayout(left_btn_layout, 3)
        top_bar.addLayout(center_layout, 4)
        top_bar.addLayout(right_btn_layout, 3)

        self.root_layout.addLayout(top_bar)

        # ===================== 中间主体区域 =====================
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # ---------- 左侧图表区 ----------
        left_layout = QGridLayout()
        left_layout.setSpacing(12)

        self.simple_flowrate_plot_view = QChartView()
        self.simple_traffic_plot_view = QChartView()
        self.simple_predict_plot_view = QChartView()
        self.simple_crosssect_plot_view = QChartView()

        for view in [
            self.simple_flowrate_plot_view,
            self.simple_traffic_plot_view,
            self.simple_predict_plot_view,
        ]:
            view.setRenderHint(QPainter.Antialiasing)
            view.setMinimumSize(500, 300)
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.simple_crosssect_plot_view.setRenderHint(QPainter.Antialiasing)
        self.simple_crosssect_plot_view.setMinimumSize(500, 300)
        self.simple_crosssect_plot_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left_layout.addWidget(self.simple_flowrate_plot_view, 0, 0)
        left_layout.addWidget(self.simple_traffic_plot_view, 0, 1)
        left_layout.addWidget(self.simple_predict_plot_view, 1, 0)
        left_layout.addWidget(self.simple_crosssect_plot_view, 1, 1)

        # ---------- 右侧视频+参数 ----------
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        self.simple_flowrate_camera = QLabel("当前采集流速画面")
        self.simple_flowrate_camera.setAlignment(QtCore.Qt.AlignCenter)
        self.simple_flowrate_camera.setStyleSheet("""
            QLabel {
                border: 2px solid #999;
                background-color: black;
                color: white;
                border-radius: 8px;
            }
        """)
        self.simple_flowrate_camera.setMinimumSize(520, 360)
        self.simple_flowrate_camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        right_layout.addWidget(self.simple_flowrate_camera, 3)

        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)
        form_layout.setHorizontalSpacing(16)
        form_layout.setVerticalSpacing(22)
        form_layout.setContentsMargins(10, 10, 10, 10)

        label_font = QFont("Microsoft YaHei", 30)
        label_font.setBold(True)

        value_font = QFont("Microsoft YaHei", 30)
        value_font.setBold(True)

        form_layout.setColumnStretch(0, 0)
        form_layout.setColumnStretch(1, 1)
        form_layout.setColumnStretch(2, 0)

        def create_row(row, title_text):
            title = QLabel(title_text)
            title.setFont(label_font)
            title.setFixedWidth(170)

            value = QLineEdit()
            value.setReadOnly(True)
            value.setFont(value_font)
            value.setMinimumHeight(74)
            value.setMaximumWidth(520)

            unit = QLabel("")
            unit.setFont(label_font)
            unit.setFixedWidth(90)
            unit.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

            form_layout.addWidget(title, row, 0)
            form_layout.addWidget(value, row, 1)
            form_layout.addWidget(unit, row, 2)

            return value, unit

        self.label_v_value, self.label_v_unit = create_row(0, "当前流速")
        self.label_q_value, self.label_q_unit = create_row(1, "当前流量")

        self.label_density_title = QLabel("密度")
        self.label_density_title.setFont(label_font)
        self.label_density_title.setFixedWidth(170)

        self.label_density_value = QDoubleSpinBox()
        self.label_density_value.setFont(value_font)
        self.label_density_value.setMinimumHeight(74)
        self.label_density_value.setMaximumWidth(520)
        self.label_density_value.setDecimals(3)
        self.label_density_value.setRange(0.001, 100.0)
        self.label_density_value.setValue(2.400)

        self.label_density_unit = QLabel("t/m³")
        self.label_density_unit.setFont(label_font)
        self.label_density_unit.setFixedWidth(90)
        self.label_density_unit.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        form_layout.addWidget(self.label_density_title, 2, 0)
        form_layout.addWidget(self.label_density_value, 2, 1)
        form_layout.addWidget(self.label_density_unit, 2, 2)

        self.label_mass_rate_value, self.label_mass_rate_unit = create_row(3, "当前质量流量")
        self.label_total_mass_value, self.label_total_mass_unit = create_row(4, "当前总质量")

        self.label_v_unit.setText("m/s")
        self.label_q_unit.setText("m³/s")
        self.label_mass_rate_unit.setText("t/min")
        self.label_total_mass_unit.setText("t")

        right_layout.addWidget(form_widget, 2, alignment=QtCore.Qt.AlignTop)

        content_layout.addLayout(left_layout, 3)
        content_layout.addLayout(right_layout, 2)

        self.root_layout.addLayout(content_layout)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.clock_timer = QtCore.QTimer(MainWindow)
        self.clock_timer.timeout.connect(self.update_current_time)
        self.clock_timer.start(1000)
        self.update_current_time()

    def update_current_time(self):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.label_current_time.setText(now_str)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Simplified MainWindow"))
        self.stop.setText(_translate("MainWindow", "结束流量检测"))
        self.start.setText(_translate("MainWindow", "开始流量检测"))
        self.return_btn.setText(_translate("MainWindow", "返回"))
        self.openprojectui.setText(_translate("MainWindow", "设置"))
        self.button_simple_savevideo.setText(_translate("MainWindow", "开始保存视频"))
        self.history_plot_btn.setText(_translate("MainWindow", "历史曲线"))
        self.label_title.setText(_translate("MainWindow", "熔渣流量在线测量系统"))