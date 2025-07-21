# ==============================================================================
#                 depth_measure.py (v10 - 简易轮廓平滑版)
# ==============================================================================

import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os
# 仅需savgol_filter，不再需要距离计算
from scipy.signal import savgol_filter

# --- 全局设置：让Matplotlib正确显示中文和负号 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#      【辅助类 RectangularSelector 】
# ==============================================================================
class RectangularSelector:
    """管理交互式绘图中的点选择 (y/n/esc 控制)。"""

    def __init__(self, ax, all_points_xy):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.all_points_xy = all_points_xy
        self.unselected_plot = ax.scatter(all_points_xy[:, 0], all_points_xy[:, 1], s=5, c='gray', alpha=0.3)
        self.selected_plot = ax.scatter([], [], s=25, c='blue', ec='k', lw=0.5)
        self.selected_mask = np.zeros(len(all_points_xy), dtype=bool)
        self.key_pressed = None
        self.rect = Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.2, edgecolor='black', lw=1, visible=False)
        self.ax.add_patch(self.rect)
        self.start_point = None
        self.connect_events()

    def connect_events(self):
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        key = event.key.lower()
        if key in ['y', 'n', 'escape']:
            self.key_pressed = key
            if key == 'y': print(f"用户按下 'y'。确认选中 {np.sum(self.selected_mask)} 个点。")
            if key == 'n':
                print("用户按下 'n'。选择已清空，请重新绘制矩形。")
                self.selected_mask.fill(False)
                self.update_plot_highlight()
                return
            if key == 'escape':
                self.selected_mask.fill(False)
                print("用户按下 'escape'。中止操作。")
            plt.close(self.ax.figure)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.start_point = (event.xdata, event.ydata)
        self.rect.set_xy(self.start_point);
        self.rect.set_width(0);
        self.rect.set_height(0)
        self.rect.set_visible(True)
        self.canvas.draw_idle()

    def on_motion(self, event):
        if self.start_point is None or event.inaxes != self.ax: return
        x0, y0 = self.start_point;
        x1, y1 = event.xdata, event.ydata
        self.rect.set_xy((min(x0, x1), min(y0, y1)))
        self.rect.set_width(abs(x0 - x1));
        self.rect.set_height(abs(y0 - y1))
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.start_point is None: return
        self.rect.set_visible(False)
        x0, y0 = self.start_point;
        x1, y1 = event.xdata, event.ydata
        self.start_point = None
        min_x, max_x = sorted([x0, x1]);
        min_y, max_y = sorted([y0, y1])
        self.selected_mask = (self.all_points_xy[:, 0] >= min_x) & (self.all_points_xy[:, 0] <= max_x) & \
                             (self.all_points_xy[:, 1] >= min_y) & (self.all_points_xy[:, 1] <= max_y)
        self.update_plot_highlight()
        print(f"选择更新完毕。请按 'y' 确认, 或 'n' 清空重选。")

    def update_plot_highlight(self):
        self.unselected_plot.set_offsets(self.all_points_xy[~self.selected_mask])
        self.selected_plot.set_offsets(self.all_points_xy[self.selected_mask])
        self.canvas.draw_idle()

    def get_final_selection(self):
        return self.selected_mask, self.key_pressed


# ==============================================================================
#      【【【 V10 - 核心修改函数：基于X轴排序的简化平滑 】】】
# ==============================================================================
def order_and_smooth_points(points_xy: np.ndarray, window_length: int = 21, polyorder: int = 3):
    """
    对描述开放轮廓的2D点集，按X轴排序并进行平滑处理。

    Args:
        points_xy (np.ndarray): 形状为 (N, 2) 的无序2D点集。
        window_length (int): Savitzky-Golay滤波器的窗口大小，必须是奇数。值越大平滑效果越强。
        polyorder (int): 滤波器拟合的多项式阶数，必须小于window_length。

    Returns:
        np.ndarray: 形状为 (N, 2) 的、经过排序和平滑处理后的点集。
    """
    # 检查输入点数是否足够进行平滑
    if len(points_xy) < window_length:
        print("警告：点数过少，无法进行平滑处理，将仅返回排序后的点。")
        # 即使不平滑，也要保证排序
        sort_indices = np.argsort(points_xy[:, 0])
        return points_xy[sort_indices]

    print("开始对轮廓点进行排序与平滑...")
    # 1. 排序：直接按X轴坐标排序
    sort_indices = np.argsort(points_xy[:, 0])
    sorted_points = points_xy[sort_indices]

    # 提取排好序的X和Y坐标
    sorted_x = sorted_points[:, 0]
    sorted_y = sorted_points[:, 1]

    # 2. 平滑：对Y坐标应用Savitzky-Golay滤波器。
    # 对于开放轮廓，不再需要复杂的填充(padding)。
    smoothed_y = savgol_filter(sorted_y, window_length, polyorder)

    # 3. 重新组合成点集
    smoothed_points = np.vstack((sorted_x, smoothed_y)).T

    print("轮廓平滑处理完成。")
    return smoothed_points


# ==============================================================================
#                  【【【 主功能函数 】】】
# ==============================================================================
def depth_measure(plane_params: np.ndarray, points_camera: np.ndarray):
    """通过稳健的GUI交互流程计算点云深度，并在框选前增加了轮廓平滑预处理。"""
    # 1. 加载和准备数据
    # try:
    #     points_camera = sio.loadmat(camera_points_path)['laser_camera'].T
    #     print(f"成功从 '{camera_points_path}' 加载了 {len(points_camera)} 个点。")
    # except Exception as e:
    #     sys.exit(f"错误: 加载点云文件 '{camera_points_path}' 失败: {e}")

    # 2. 旋转点云 (逻辑不变)
    V = plane_params[:3];
    V_norm = V / np.linalg.norm(V)
    VEC_0 = np.array([0, 0, 1])
    theta = np.arccos(np.clip(np.dot(VEC_0, V_norm), -1.0, 1.0))
    if theta > np.pi / 2: theta = np.pi - theta; V_norm = -V_norm
    rot_axis = np.cross(V_norm, VEC_0)
    if np.linalg.norm(rot_axis) < 1e-8:
        R_mat = np.eye(3) if np.allclose(V_norm, VEC_0) else -np.eye(3)
    else:
        rot_axis /= np.linalg.norm(rot_axis)
        ax_r, ay_r, az_r = rot_axis
        rot_axis_skew = np.array([[0, -az_r, ay_r], [az_r, 0, -ax_r], [-ay_r, ax_r, 0]])
        R_mat = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(rot_axis, rot_axis) + np.sin(
            theta) * rot_axis_skew
    points_rotated = (R_mat @ points_camera.T).T
    points_rotated_xy = points_rotated[:, :2]

    # 3. 【V10 优化步骤】对2D点进行排序和平滑
    smoothed_contour_xy = order_and_smooth_points(points_rotated_xy, window_length=21, polyorder=3)

    # 4. 计算显示范围
    x_min, x_max = points_rotated_xy[:, 0].min(), points_rotated_xy[:, 0].max()
    y_min, y_max = points_rotated_xy[:, 1].min(), points_rotated_xy[:, 1].max()
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    plot_dim = max(x_max - x_min, y_max - y_min) * 1.1
    view_lims = {
        'xlim': (x_center - plot_dim / 2, x_center + plot_dim / 2),
        'ylim': (y_center - plot_dim / 2, y_center + plot_dim / 2)
    }

    # 5. 【交互步骤一】 选择基准点
    highlighted_points = None
    while True:
        fig_base, ax_base = plt.subplots(figsize=(11, 9))
        title_str = "【步骤1】选择基准点\n按 'y' 确认 | 'n' 重选 | 'esc' 或关闭窗口中止"
        ax_base.set_title(title_str)
        ax_base.plot(smoothed_contour_xy[:, 0], smoothed_contour_xy[:, 1], 'b-', lw=1.5, label='平滑后轮廓')

        selector_base = RectangularSelector(ax_base, points_rotated_xy)
        ax_base.set_xlabel('X (俯视图)');
        ax_base.set_ylabel('Y (俯视图)')
        ax_base.grid(True);
        ax_base.set_aspect('equal', adjustable='box');
        ax_base.set(**view_lims)
        ax_base.legend([selector_base.unselected_plot, selector_base.selected_plot, ax_base.lines[0]],
                       ['原始点', '当前选中点', '平滑后轮廓'], loc='upper right')

        print("\n--> 请在弹出的窗口中选择“基准点”...")
        plt.show()

        selection_mask, key_pressed = selector_base.get_final_selection()
        if key_pressed == 'y':
            if np.sum(selection_mask) < 2:
                print(f"\n错误: 拟合直线至少需要2个点，请重试。")
                continue
            highlighted_points = points_rotated[selection_mask]
            break
        else:
            print("\n用户取消了基准点选择，程序终止。")
            return None

    # 6. 拟合基准线
    model = LinearRegression().fit(highlighted_points[:, 0].reshape(-1, 1), highlighted_points[:, 1])
    k, b = model.coef_[0], model.intercept_
    print(f"\n基准线拟合结果: y = {k:.4f} * x + {b:.4f}")

    # 7. 【交互步骤二】 选择测量点 (后续流程无改动)
    selected_points = None
    while True:
        fig_depth, ax_depth = plt.subplots(figsize=(11, 9))
        title_str_2 = "【步骤2】选择测量点\n按 'y' 确认 | 'n' 重选 | 'esc' 或关闭窗口中止"
        ax_depth.set_title(title_str_2)
        ax_depth.plot(smoothed_contour_xy[:, 0], smoothed_contour_xy[:, 1], 'b-', lw=1.5, label='平滑后轮廓')
        ax_depth.scatter(highlighted_points[:, 0], highlighted_points[:, 1], s=15, c='red', label='已确认的基准点')

        selector_depth = RectangularSelector(ax_depth, points_rotated_xy)
        ax_depth.set_xlabel('X (俯视图)');
        ax_depth.set_ylabel('Y (俯视图)')
        ax_depth.grid(True);
        ax_depth.set_aspect('equal', adjustable='box');
        ax_depth.set(**view_lims)
        handles, labels = ax_depth.get_legend_handles_labels()
        handles.extend([selector_depth.unselected_plot, selector_depth.selected_plot])
        labels.extend(['原始点', '当前选中测量点'])
        ax_depth.legend(handles, labels, loc='upper right')

        print("\n--> 请在弹出的新窗口中选择“测量点”...")
        plt.show()

        selection_mask, key_pressed = selector_depth.get_final_selection()
        if key_pressed == 'y':
            if np.sum(selection_mask) == 0:
                print("\n错误: 您没有选择任何测量点，请重试。")
                continue
            selected_points = points_rotated[selection_mask]
            break
        else:
            print("\n用户取消了测量点选择，程序终止。")
            return None

    # 8. 计算深度
    depth_values = np.abs(k * selected_points[:, 0] - selected_points[:, 1] + b) / np.sqrt(k ** 2 + 1)
    print("\n--- 计算结果 ---")
    for i, d in enumerate(depth_values): print(f"  采样点 {i + 1}: 深度 = {d:.4f}")

    # 9. 最终结果可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(highlighted_points[:, 0], highlighted_points[:, 1], c='red', label='基准线点')
    plt.scatter(selected_points[:, 0], selected_points[:, 1], c='green', marker='^', s=60, label='深度采样点')
    x_lim_final = plt.gca().get_xlim()
    x_range_final = np.linspace(x_lim_final[0], x_lim_final[1], 200)
    plt.plot(x_range_final, k * x_range_final + b, 'r-', label=f'拟合基准线 y={k:.2f}x+{b:.2f}')
    plt.title('最终深度测量结果');
    plt.xlabel('X (俯视图)');
    plt.ylabel('Y (俯视图)')
    plt.legend();
    plt.grid(True);
    plt.gca().set_aspect('equal', adjustable='box');
    plt.gca().set(**view_lims)
    print("\n显示最终结果图。关闭该图后程序将完全结束。")
    plt.show()

    return depth_values


# ==============================================================================
#                                  主程序入口
# ==============================================================================
if __name__ == '__main__':
    CAMERA_POINTS_FILE = 'curve_data_InCamera.mat'
    LASER_PLANE_FILE = 'planeParams_vertical_goucao3.npy'

    print("=" * 70)
    print("      独立运行: 测量三维点云高度差 (v10 - 基于X轴排序平滑)")
    print("=" * 70)

    if not all([os.path.exists(f) for f in [CAMERA_POINTS_FILE, LASER_PLANE_FILE]]):
        sys.exit(f"错误: 必需文件缺失。请确保 '{CAMERA_POINTS_FILE}' 和 '{LASER_PLANE_FILE}' 都在当前目录下。")
    try:
        laser_plane_parameters = np.load(LASER_PLANE_FILE)
    except Exception as e:
        sys.exit(f"错误: 加载激光平面参数文件 '{LASER_PLANE_FILE}' 失败: {e}")

    depth_results = depth_measure(laser_plane_parameters, CAMERA_POINTS_FILE)

    if depth_results is not None:
        print("\n深度测量流程成功完成。")
