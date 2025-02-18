#「3d视觉工坊」：目前国内最大的3D视觉知识交流圈，致力于传播最前沿的3d视觉领域知识！
# 作者：曹博
# 邮箱：fly_cjb@163.com
import os
import cv2
import numpy as np
import matplotlib.pylab as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


class Calibrator:
    def __init__(self):
        self.img_size = None        # 图像尺寸（H, W）
        self.points_world_xyz = []  # 世界坐标
        self.points_pixel_xy  = []  # 像素坐标
        self.error = None           # 重投影误差
        self.mtx   = None           # 内参矩阵：k1、k2、p1、p2、k3
        self.dist  = None           # 畸变系数
        self.rvecs = None           # 旋转矩阵
        self.tvecs = None           # 平移矩阵
        self.calib_info = {}

    def detect(self, cols, rows, folder, show):
        assert ((cols > 0) & (rows > 0))  # 逻辑运算＋括号，指定运算顺序
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 停止迭代的标准

        # 角点的世界坐标
        point_world_xyz = np.zeros((rows * cols, 3), np.float32)
        point_world_xyz[:, :2] = np.mgrid[0: cols, 0: rows].T.reshape(-1, 2) * 10

        # np.mgrid[start:end:step]
        # start:开始坐标
        # end:结束坐标（实数不包括，复数包括）
        # step:步长
        # 2D结构:x,y=np.mgrid[-5:5:3j,-2:2:3j]
        # 其中x沿着水平向右的方向扩展(即是：每列都相同)，y沿着垂直的向下的方向扩展(即是：每行都相同)。
        # >>> x
        # array([[-5., -5., -5.],
        #        [ 0.,  0.,  0.],
        #        [ 5.,  5.,  5.]])
        # >>> y
        # array([[-2.,  0.,  2.],
        #        [-2.,  0.,  2.],
        #        [-2.,  0.,  2.]])

        # 标定的文件
        calib_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".jpg")]
        calib_files.sort()  # 内部进行排序
        for filename in calib_files:
            img = self.imread(filename)
            if img is None:
                raise FileNotFoundError(filename, "没有发现!")
            if len(img.shape) == 2:
                gray = img
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.img_size is None:
                self.img_size = gray.shape[::-1]
            else:
                assert gray.shape[::-1] == self.img_size

            #  01 角点粗检测
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                # 02 角点精检测
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                print(corners)
                # 03 添加角点像素坐标、世界坐标
                self.points_pixel_xy.append(corners)
                
                self.points_world_xyz.append(point_world_xyz)
            else:
                print("未检测到角点:", filename)
            if show:
                if len(img.shape) == 2:
                    print(img.shape)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                title = os.path.basename(filename)
                cv2.imshow(title, img)
                cv2.moveWindow(title, 500, 200)
                cv2.waitKey(0)

    def calib(self):
        # 03 标定
        self.error, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.points_world_xyz,  # 世界坐标
            self.points_pixel_xy,   # 像素坐标
            self.img_size,          # 图像尺寸
            None, None
        )

    # 校正+裁剪：怎么用？之后相机、世界坐标之间怎么转换，还有跟一些IMU怎么融合啊等等问题！
    # 我实在不建议自己从零去实现一个代码，因为太耗费时间了，3D视觉很多算法太复杂了，自己去一个个事无巨细去实现，不太可能（当然原理你得懂嘛）
    # 建议从开源代码开始，看人家怎么用，在它基础上改就好了，解决你自己的问题
    def rectify(self, img):
        # 04 使用   mtx： 相机内参矩阵 dist: 畸变系数矩阵
        return cv2.undistort(img, self.mtx, self.dist)

    def save_calib_info(self, save_file):
        # 效果好坏评价
        print("重投影误差:\n", self.error, "\n")
        print("相机内参:\n", self.mtx, "\n")
        print("相机畸变:\n", self.dist, "\n")
        # 每个标定板（左上角角点）相对于相机原点的平移、旋转矩阵
        print("旋转矩阵:\n", self.rvecs, "\n")
        print("平移矩阵:\n", self.tvecs, "\n")
        self.calib_info["Error"] = self.error
        self.calib_info["Ac"] = self.mtx
        self.calib_info["Dist"] = self.dist
        self.calib_info["Rs"] = self.rvecs
        self.calib_info["Ts"] = self.tvecs
        np.save(save_file, self.calib_info)
        print("保存标定信息到文件:",  save_file)

    # 仅支持Windows中文路径图片读取
    @staticmethod
    def imread(filename: str):
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)


if __name__ == '__main__':
    ############### 超参数 ###############
    folder = r"./new_imgs"
    save_file = r"./calib_info.npy"
    show = True
    cols = 11  # 有多少列角点
    rows = 8  # 有多少行角点

    ############### 标定 ###############
    calibrator = Calibrator()
    # 01 检测角点
    calibrator.detect(cols, rows, folder, show)
    # 02 标定相机
    calibrator.calib()
    # 03 保存标定
    calibrator.save_calib_info(save_file)
    # 04 校正图片
    img_test = calibrator.imread(os.path.join(folder, "1.jpg"))
    dst1 = calibrator.rectify(img_test)
    # plt.imshow(img_test, cmap='gray'), plt.title("原图"), plt.show()
    # plt.imshow(dst1, cmap='gray'), plt.title("校正"), plt.show()
    # plt.imshow(dst1 - img_test, cmap='gray'), plt.title("差别"), plt.show()
