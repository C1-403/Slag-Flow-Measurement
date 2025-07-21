# ==============================================================================
#                 project_and_smooth.py
#          功能: 将3D点云投影为2D轮廓，并进行排序平滑处理
# ==============================================================================

import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import sys
import os

# --- 全局设置：让Matplotlib正确显示中文和负号 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#      【核心函数：基于X轴排序的简化平滑 】
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
    if len(points_xy) < window_length:
        print("警告：点数过少，无法进行平滑处理，将仅返回按X轴排序后的点。")
        sort_indices = np.argsort(points_xy[:, 0])
        return points_xy[sort_indices]

    print("开始对轮廓点进行排序与平滑...")
    # 1. 排序：直接按X轴坐标排序
    sort_indices = np.argsort(points_xy[:, 0])
    sorted_points = points_xy[sort_indices]

    # 提取排好序的X和Y坐标
    sorted_x = sorted_points[:, 0]
    sorted_y = sorted_points[:, 1]

    # 2. 平滑：对Y坐标应用Savitzky-Golay滤波器
    smoothed_y = savgol_filter(sorted_y, window_length, polyorder)

    # 3. 重新组合成点集
    smoothed_points = np.vstack((sorted_x, smoothed_y)).T

    print("轮廓平滑处理完成。")
    return smoothed_points


# ==============================================================================
#                  【【【 新功能主函数 】】】
# ==============================================================================
def project_and_smooth(plane_params: np.ndarray, points_camera: np.ndarray, show_plot: bool = False):
    """
    加载3D点云，将其投影到2D平面，然后对2D轮廓进行排序和平滑。

    Args:
        plane_params (np.ndarray): 激光平面的参数 [A, B, C, D]。
        points_camera (np.ndarray): 相机系激光点3D坐标。

    Returns:
        np.ndarray: 形状为 (N, 2) 的平滑后的2D点集，如果失败则返回 None。
    """
    # 1. 计算旋转矩阵，将激光平面旋转至与XY平面平行
    V = plane_params[:3]
    V_norm = V / np.linalg.norm(V)
    VEC_0 = np.array([0, 0, 1])  # 目标法向量，即Z轴

    theta = np.arccos(np.clip(np.dot(VEC_0, V_norm), -1.0, 1.0))
    if theta > np.pi / 2:
        theta = np.pi - theta
        V_norm = -V_norm

    rot_axis = np.cross(V_norm, VEC_0)
    if np.linalg.norm(rot_axis) < 1e-8:
        R_mat = np.eye(3) if np.allclose(V_norm, VEC_0) else -np.eye(3)
    else:
        rot_axis /= np.linalg.norm(rot_axis)
        ax_r, ay_r, az_r = rot_axis
        rot_axis_skew = np.array([[0, -az_r, ay_r], [az_r, 0, -ax_r], [-ay_r, ax_r, 0]])
        R_mat = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(rot_axis, rot_axis) + np.sin(
            theta) * rot_axis_skew

    # 2. 应用旋转，并提取2D坐标
    points_rotated = (R_mat @ points_camera.T).T
    points_rotated_xy = points_rotated[:, :2]

    # 3. 对2D点进行排序和平滑
    smoothed_contour_xy = order_and_smooth_points(points_rotated_xy, window_length=21, polyorder=3)

    # 4. 可视化结果 先不显示了
    if show_plot:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.scatter(points_rotated_xy[:, 0], points_rotated_xy[:, 1], c='gray', s=5, alpha=0.6, label='原始投影点')
        ax.plot(smoothed_contour_xy[:, 0], smoothed_contour_xy[:, 1], 'r-', lw=2, label='平滑后轮廓')
        ax.set_title('3D点云投影与平滑结果')
        ax.set_xlabel('X (俯视图)')
        ax.set_ylabel('Y (俯视图)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        print("\n显示投影平滑结果图。关闭窗口后程序将继续。")
        plt.show()

    # 6. 返回处理好的二维点集
    return smoothed_contour_xy


# ==============================================================================
#                                  主程序入口
# ==============================================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    import os
    import sys

    # --- 1. 配置文件路径 ---
    CAMERA_POINTS_FILE = 'curve_data_InCamera.mat' # 这是包含3D坐标的文件
    LASER_PLANE_FILE = 'planeParams_vertical_goucao3.npy'

    print("="*70)
    print("      运行: 3D点云投影平滑工具")
    print("="*70)

    # --- 2. 检查并加载所有输入文件 ---
    if not all([os.path.exists(f) for f in [CAMERA_POINTS_FILE, LASER_PLANE_FILE]]):
        sys.exit(f"错误: 必需文件缺失。请确保文件都在当前目录下。")

    try:
        laser_plane_parameters = np.load(LASER_PLANE_FILE)
        print(f"成功从 '{LASER_PLANE_FILE}' 加载激光平面参数。")
    except Exception as e:
        sys.exit(f"错误: 加载激光平面参数文件失败: {e}")

    # 【修正点】在这里从.mat文件加载3D点云数据
    try:
        # 注意：MATLAB中保存的变量名是 'laser_camera'，并且是 (3, N) 格式
        points_camera_3d = sio.loadmat(CAMERA_POINTS_FILE)['laser_camera'].T
        print(f"成功从 '{CAMERA_POINTS_FILE}' 加载了 {len(points_camera_3d)} 个3D点。")
    except Exception as e:
        sys.exit(f"错误: 加载相机坐标点云文件失败: {e}")


    # --- 3. 调用核心功能 ---
    # 【修正点】将加载好的Numpy数组 `points_camera_3d` 传递给函数
    smoothed_points = project_and_smooth(
        plane_params=laser_plane_parameters,
        points_camera=points_camera_3d  # <--- 现在传递的是正确的数组对象
    )

    if smoothed_points is not None:
        print(f"\n处理成功！返回了 {len(smoothed_points)} 个平滑后的2D点。")
        # 可以选择将结果保存
        # np.save('smoothed_contour.npy', smoothed_points)