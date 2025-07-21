import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- 导入项目中的各个功能模块 ---
from load_camera_params import load_matlab_camera_params
from extract_laser import extract_laser_points
from laser_plane_fitting import calculate_laser_plane
from extract_subject import extract_subject  # 导入您重命名后的物体激光提取函数
from laser_construction import reconstruct_laser_line  # 导入三维重建函数
from depth_measure import depth_measure
# --- 全局设置：让Matplotlib正确显示中文和负号 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
#                                主程序入口
# ==============================================================================
if __name__ == '__main__':
    # ==========================================================================
    #             第一部分：激光平面标定
    # ==========================================================================
    HAS_CALIBRATED = 1
    # --- 步骤 0: 加载与设定 ---
    # CALIB_DATA_PATH = '20250523'
    # CALIB_IMAGE_FILENAMES = ['1.png', '2.png', '3.png']
    # # 【重要】Python索引 (MATLAB中的索引值减1)
    # LASER_IMAGE_INDICES = [35, 36, 37]
    # CAMERA_PARAMS_MAT_FILE = 'camera_params_for_python.mat'
    # PLANE_PARAMS_OUTPUT_FILE = 'planeParams_vertical_goucao3'  # 输出文件名(无后缀)
    # plane_params_npy_file = PLANE_PARAMS_OUTPUT_FILE + '.npy'

    CALIB_DATA_PATH = '20250517'
    CALIB_IMAGE_FILENAMES = ['1.png', '2.png', '3.png']
    # 【重要】Python索引 (MATLAB中的索引值减1)
    LASER_IMAGE_INDICES = [45, 46, 47]
    CAMERA_PARAMS_MAT_FILE = 'camera_params_for_python.mat'
    PLANE_PARAMS_OUTPUT_FILE = 'planeParams_vertical_goucao2'  # 输出文件名(无后缀)
    PLANE_PARAMS_NPY_FILE = PLANE_PARAMS_OUTPUT_FILE + '.npy'
    if HAS_CALIBRATED:
        camera_parameters = load_matlab_camera_params(CAMERA_PARAMS_MAT_FILE)
    else:
        print("=" * 60)
        print("                第一部分: 激光平面方程标定")
        print("=" * 60)

        print("--- 步骤 1/3: 加载相机参数 ---")
        camera_parameters = load_matlab_camera_params(CAMERA_PARAMS_MAT_FILE)

        if not camera_parameters:
            sys.exit("\n主程序终止：因相机参数加载失败，无法继续执行。")

        # --- 步骤 2/3: 从标定板图像中交互式提取激光点 ---
        print("\n--- 步骤 2/3: 提取标定板激光点 ---")
        all_laser_points_list = []
        point_counts = []

        for filename in CALIB_IMAGE_FILENAMES:
            full_path = os.path.join(CALIB_DATA_PATH, filename)
            if not os.path.exists(full_path):
                print(f"错误：找不到标定图像文件 {full_path}，跳过此文件。")
                continue

            # 调用标定板的激光提取函数
            points = extract_laser_points(
                image_path=full_path,
                camera_matrix=camera_parameters['intrinsic_matrix'],
                dist_coeffs=camera_parameters['dist_coeffs']
            )
            if points is not None and len(points) > 0:
                all_laser_points_list.append(points)
                point_counts.append(len(points))
            else:
                print(f"在图像 {filename} 上的操作被取消或未能提取到点。")

        # --- 步骤 3/3: 解算并保存激光平面方程 ---

        # 仅当成功提取到点时才进行平面拟合
        if all_laser_points_list:
            all_points_2d_array = np.vstack(all_laser_points_list)
            print("\n--- 步骤 3/3: 解算激光平面方程 ---")
            calculate_laser_plane(
                laser_indices=LASER_IMAGE_INDICES,
                point_counts=point_counts,
                all_extracted_points=all_points_2d_array,
                camera_params=camera_parameters,
                output_filename=PLANE_PARAMS_OUTPUT_FILE  # 传入无后缀的文件名
            )
        else:
            print("\n警告：未能从任何标定图像中提取到激光点，无法进行新的平面拟合。")
            # 检查是否已有旧的平面文件，如果没有则无法继续
            if not os.path.exists(PLANE_PARAMS_NPY_FILE):
                sys.exit("程序终止，因为没有提取到新的点，且不存在旧的平面参数文件。")

    # ==========================================================================
    #               第二部分：物体三维轮廓重建 (新增集成功能)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("                第二部分: 物体表面三维重建")
    print("=" * 60)

    # 检查平面参数文件是否已成功生成或原已存在
    if not os.path.exists(PLANE_PARAMS_NPY_FILE):
        print(f"错误：激光平面参数文件 '{PLANE_PARAMS_NPY_FILE}' 未找到。")
        sys.exit("无法继续进行三维重建。请先确保第一部分的平面标定成功。")

    print(f"已检测到激光平面参数文件: '{PLANE_PARAMS_NPY_FILE}'")


    # --- 步骤 4/5: 提取物体表面的2D激光线 ---
    print("\n--- 步骤 4/5: 提取物体表面激光线 (2D) ---")
    SUBJECT_IMAGE_FILE = '20250523/Pic_20250523152209308.png'
    UV_OUTPUT_FILE = 'uv.mat'

    if not os.path.exists(SUBJECT_IMAGE_FILE):
        sys.exit(f"错误: 找不到目标物体图像 '{SUBJECT_IMAGE_FILE}'，程序终止。")

    # 调用 extract_subject 函数, 它会处理交互和保存 uv.mat 的过程
    # camera_parameters 是第一部分加载的，可直接复用
    image_data = cv2.imread(SUBJECT_IMAGE_FILE)
    uv_points,_ = extract_subject(
        image_data=image_data,
        camera_params=camera_parameters
    )

    # --- 步骤 5/5: 根据2D点重建3D轮廓 ---
    laser_plane_parameters = np.load(PLANE_PARAMS_NPY_FILE)
    points_camera = reconstruct_laser_line(
        camera_params=camera_parameters,
        plane_params=laser_plane_parameters,  # <--- 传入 NumPy 数组
        subject_uv=uv_points
    )

    print("=" * 60)
    print("                独立运行: 测量三维点云高度差")
    print("=" * 60)
    # CAMERA_POINTS_FILE = 'curve_data_InCamera.mat'
    # if not all([os.path.exists(f) for f in [CAMERA_POINTS_FILE, PLANE_PARAMS_NPY_FILE]]):
    #     print(f"错误: 必需文件缺失。请确保 '{CAMERA_POINTS_FILE}' 和 '{PLANE_PARAMS_NPY_FILE}' 都在当前目录下。")
    #     sys.exit()
    #
    # try:
    #     laser_plane_parameters = np.load(PLANE_PARAMS_NPY_FILE)
    #     print(f"成功从 '{PLANE_PARAMS_NPY_FILE}' 加载激光平面参数。")
    # except Exception as e:
    #     sys.exit(f"错误: 加载激光平面参数文件 '{PLANE_PARAMS_NPY_FILE}' 失败: {e}")

    depth_results = depth_measure(laser_plane_parameters, points_camera)

    if depth_results is not None:
        print("\n深度测量流程成功完成。")

