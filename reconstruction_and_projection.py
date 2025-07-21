import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
# --- 导入项目中的各个功能模块 ---
from extract_subject import extract_subject  # 导入您重命名后的物体激光提取函数
from laser_construction import reconstruct_laser_line  # 导入三维重建函数
from project_and_smooth import project_and_smooth
# --- 全局设置：让Matplotlib正确显示中文和负号 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def reconstruction_and_projection(image_path: str, camera_params: dict, plane_params: np.ndarray):
    # ==========================================================================
    #               物体三维轮廓重建
    # ==========================================================================
    # --- 步骤 1: 提取物体表面的2D激光线 ---
    print("\n--- 步骤 4/5: 提取物体表面激光线 (2D) ---")
    SUBJECT_IMAGE_FILE = image_path
    IMAGE_DATA = cv2.imread(image_path)
    if not os.path.exists(SUBJECT_IMAGE_FILE):
        sys.exit(f"错误: 找不到目标物体图像 '{SUBJECT_IMAGE_FILE}'，程序终止。")

    # 调用 extract_subject 函数, 它会处理交互和保存 uv.mat 的过程
    # camera_parameters 是第一部分加载的，可直接复用
    points2D_uv,_ = extract_subject(
        image_data=IMAGE_DATA,
        camera_params=camera_params
    )

    # --- 步骤 2: 根据2D点重建3D轮廓 ---

    laser_plane_parameters = plane_params

    points3D_in_camera_coords = reconstruct_laser_line(
        camera_params=camera_params,
        plane_params=laser_plane_parameters,  # <--- 传入 NumPy 数组
        subject_uv=points2D_uv
    )
    print("\n所有流程执行完毕！")

    # ==========================================================================
    #               重投影以及平滑
    # ==========================================================================
    print("=" * 60)
    print("      运行: 3D点云投影及平滑(project_and_smooth)")
    print("=" * 60)
    OUTPUT_FILE = 'smoothed_contour.npy'  # 定义输出文件名
    CAMERA_POINTS_FILE = 'curve_data_InCamera.mat'


    smoothed_points_2d = project_and_smooth(laser_plane_parameters, points3D_in_camera_coords)

    # 处理并保存结果
    if smoothed_points_2d is not None:
        # 将结果保存到文件，以便其他程序使用
        np.save(OUTPUT_FILE, smoothed_points_2d)
        print(f"\n处理成功！已将 {len(smoothed_points_2d)} 个点的平滑轮廓保存到 '{OUTPUT_FILE}'")
        return smoothed_points_2d
    else:
        print("\n处理流程未成功完成。")
        return None


# ==============================================================================
#                  【【【 计算轮廓间面积 】】】
# ==============================================================================
def calculate_area_between_contours(upper_contour: np.ndarray, lower_contour: np.ndarray, num_points: int = 2000):
    """
    使用数值积分计算两个2D轮廓线上、下包裹的面积。
    【已更新】：同时返回用于可视化的插值数据。

    Args:
        upper_contour (np.ndarray): 上轮廓的点集 (N, 2)，已按X轴排序。
        lower_contour (np.ndarray): 下轮廓的点集 (M, 2)，已按X轴排序。
        num_points (int): 用于插值的采样点数量。

    Returns:
        tuple: 一个元组，包含：
               - float: 计算出的面积值。
               - np.ndarray: 用于插值的公共X轴坐标 (num_points,)。
               - np.ndarray: 在公共X轴上插值后的上轮廓Y坐标 (num_points,)。
               - np.ndarray: 在公共X轴上插值后的下轮廓Y坐标 (num_points,)。
    """
    print("\n开始计算上下轮廓间的面积...")
    x_min = max(upper_contour[:, 0].min(), lower_contour[:, 0].min())
    x_max = min(upper_contour[:, 0].max(), lower_contour[:, 0].max())

    if x_min >= x_max:
        print("错误：上下轮廓的X轴范围没有重叠区域。", file=sys.stderr)
        return 0.0, None, None, None

    # 创建一个高密度的、共同的X轴坐标点
    common_x = np.linspace(x_min, x_max, num_points)

    # 对上下轮廓进行插值，得到长度相同的Y坐标数组
    upper_y_interp = np.interp(common_x, upper_contour[:, 0], upper_contour[:, 1])
    lower_y_interp = np.interp(common_x, lower_contour[:, 0], lower_contour[:, 1])

    # 使用梯形法则计算面积
    area = np.trapz(upper_y_interp - lower_y_interp, common_x)
    print(f"面积计算完成。")

    # 【核心修改】返回面积和用于绘图的、长度一致的数组
    return area, common_x, upper_y_interp, lower_y_interp
