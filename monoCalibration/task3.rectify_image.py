#「3d视觉工坊」：目前国内最大的3D视觉知识交流圈，致力于传播最前沿的3d视觉领域知识！
# 作者：曹博
# 邮箱：fly_cjb@163.com

import cv2
import numpy as np
import matplotlib.pylab as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


def rectify(img, Ac, dist):
    h, w = img.shape
    fx, fy = Ac[0, 0], Ac[1, 1]
    cx, cy = Ac[0, 2], Ac[1, 2]
    k1, k2, p1, p2, k3 = dist[0, :]

    img_correct = np.zeros((h, w), np.uint8)
    
    # 遍历所有(U,V)填充为映射对应的(Ud,Vd)即可实现图像去畸变处理。
    for v in range(h):
        for u in range(w):
            # 01 映射回归一化平面
            x = (u - cx) / fx
            y = (v - cy) / fy
            # 02 找到归一化平面下相应的畸变坐标（畸变校正）
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            x_distorted = x * (1 + k1 * r2 + k2 * r4 + k3 * r6) \
                          + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            y_distorted = y * (1 + k1 * r2 + k2 * r4 + k3 * r6) \
                          + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
            # 03 将畸变坐标 映射回 正确的像素坐标系（最近邻差值）
            u_distorted = round(fx * x_distorted + cx)
            v_distorted = round(fy * y_distorted + cy)
            # 04 建立映射关系：理想坐标 -> 畸变坐标
            if ((0 <= v_distorted <= h) & (0 <= u_distorted  <= w)):
                img_correct[v, u] = img[v_distorted, u_distorted]
            else:
                img_correct[v, u] = 0
    return img_correct


if __name__ == '__main__':
    filename = "imgs/1.jpg"
    img = cv2.imread(filename, flags=0)
    # 01 加载标定结果
    calib_info_file = "calib_info.npy"
    calib_info = np.load(calib_info_file, allow_pickle=True).item()
    Ac = calib_info["Ac"]  # AC相机内参
    # OpenCV:k1, k2, p1, p2, k3
    # Matlab:k1, k2, k3, p1, p2
    Dist = calib_info["Dist"]  # 相机畸变

    # 02 校正图片
    dst1 = cv2.undistort(img, Ac, Dist)  # OpenCV自带的校正函数
    dst2 = rectify(img, Ac, Dist)        # 手动实现校正函数

    # 03 展示标定结果
    dif1 = dst1 - img
    dif2 = dst2 - img
    dif3 = dst1 - dst2

    plt.imshow(img, cmap='gray'), plt.title("原图"), plt.show()
    plt.imshow(dst1, cmap='gray'), plt.title("系统校正"), plt.show()
    plt.imshow(dst2, cmap='gray'), plt.title("手动校正"), plt.show()
    plt.imshow(dif1, cmap='gray'), plt.title("差异（原图-系统）"), plt.show()
    plt.imshow(dif2, cmap='gray'), plt.title("差异（原图-手动）"), plt.show()
    # 备注：自己实现的跟OpenCV实现的会有所差异，我们校正的图存在图像模糊现象
    # 原因：OpenCV使用更为复杂的实现（双线性插值）
    plt.imshow(dif3, cmap='gray'), plt.title("差异（系统-手动）"), plt.show()
