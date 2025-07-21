# 文件名: matplotlib_roi_selector.py

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

# 预先设置好兼容的后端，以增加稳定性
import matplotlib

matplotlib.use('TkAgg')


class ROISelector:
    """
    一个使用 Matplotlib 的 RectangleSelector 来获取 ROI 的类。
    """

    def __init__(self, ax):
        self.ax = ax
        self.roi = None  # 用于存储最终的 ROI (x, y, w, h)

        # 创建 RectangleSelector
        self.selector = RectangleSelector(
            ax,
            self,  # 将类实例本身作为回调
            useblit=True,
            button=[1],  # 仅响应鼠标左键
            minspanx=5, minspany=5,  # 最小选择区域
            spancoords='pixels',
            interactive=True
        )

    def __call__(self, eclick, erelease):
        """当用户完成一次拖拽选择时，此方法会被调用"""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # 计算并保存为 (x, y, w, h) 格式
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x1 - x2)
        height = abs(y1 - y2)
        self.roi = (x, y, width, height)


def select_roi_with_matplotlib(img: np.ndarray):
    """
    使用 Matplotlib 窗口来让用户选择一个矩形 ROI。
    这是 cv2.selectROI 的直接替代品。

    Args:
        img (np.ndarray): 要显示并进行选择的图像 (灰度图或BGR图)。

    Returns:
        tuple: (x, y, w, h) 格式的ROI，如果未选择则返回 (0, 0, 0, 0)。
    """
    print("\n" + "=" * 60)
    print("请在弹出的【Matplotlib窗口】中用鼠标拖拽一个矩形框。")
    print("选择完毕后，【关闭该窗口】即可继续程序。")
    print("=" * 60)

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))

    # 根据图像是彩色还是灰度来显示
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        # OpenCV 使用 BGR，Matplotlib 使用 RGB，需要转换
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax.set_title("请选择ROI, 然后关闭此窗口")

    # 实例化我们的选择器
    selector_obj = ROISelector(ax)

    # 显示窗口，并阻塞程序直到窗口被关闭
    plt.show(block=True)

    # 窗口关闭后，返回存储的ROI
    if selector_obj.roi:
        print(f"ROI 已选择: {selector_obj.roi}")
        return selector_obj.roi
    else:
        print("未选择任何ROI。")
        return (0, 0, 0, 0)
