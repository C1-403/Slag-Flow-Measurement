import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.io as sio

# --- 导入项目中的各个功能模块 ---
from load_camera_params import load_matlab_camera_params
from extract_laser import extract_laser_points
from laser_plane_fitting import calculate_laser_plane
from extract_subject import extract_subject  # 导入您重命名后的物体激光提取函数
from laser_construction import reconstruct_laser_line  # 导入三维重建函数
from project_and_smooth import project_and_smooth
from reconstruction_and_projection import reconstruction_and_projection
from reconstruction_and_projection import calculate_area_between_contours
from area_calculator import *
from configuration_and_calbration import load_config_from_file
# --- 全局设置：让Matplotlib正确显示中文和负号 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def van(UPPER_IMAGE_TO_TEST, HAS_CALIBRATED=True):
    # ==========================================================================
    #             第一部分：激光平面标定
    # ==========================================================================
    # --- 步骤 0: 加载与设定 ---

    # CALIB_DATA_PATH = '20250523'
    # CALIB_IMAGE_FILENAMES = ['1.png', '2.png', '3.png']
    # # 【重要】Python索引 (MATLAB中的索引值减1)
    # LASER_IMAGE_INDICES = [35, 36, 37]
    # CAMERA_PARAMS_MAT_FILE = 'camera_params_for_python.mat'
    # PLANE_PARAMS_OUTPUT_FILE = 'planeParams_vertical_goucao3'  # 输出文件名(无后缀)
    # PLANE_PARAMS_NPY_FILE = PLANE_PARAMS_OUTPUT_FILE + '.npy'
    # UPPER_SUBJECT_IMAGE_FILE = '20250523/Pic_20250523152209308.png'
    # LOWER_SUBJECT_IMAGE_FILE = '20250523/Pic_20250523152227835.png'

    config_data = load_config_from_file('config.txt')
    # 检查配置是否加载成功
    if not config_data:
        print("脚本因配置加载失败而退出。")
    else:
        # 你现在可以像使用字典一样使用这些参数
        CALIB_DATA_PATH = config_data.get('CALIB_DATA_PATH')
        CALIB_IMAGE_FILENAMES = config_data.get('CALIB_IMAGE_FILENAMES')
        LASER_IMAGE_INDICES = config_data.get('LASER_IMAGE_INDICES')
        CAMERA_PARAMS_MAT_FILE = config_data.get('CAMERA_PARAMS_MAT_FILE')
        PLANE_PARAMS_OUTPUT_FILE = config_data.get('PLANE_PARAMS_OUTPUT_FILE')
        PLANE_PARAMS_NPY_FILE = PLANE_PARAMS_OUTPUT_FILE + '.npy'
        #举例
        # CALIB_DATA_PATH = '20250517'
        # CALIB_IMAGE_FILENAMES = ['1.png', '2.png', '3.png']
        # # 【重要】Python索引 (MATLAB中的索引值减1)
        # LASER_IMAGE_INDICES = [45, 46, 47]
        # CAMERA_PARAMS_MAT_FILE = 'camera_params_for_python.mat'
        # PLANE_PARAMS_OUTPUT_FILE = 'planeParams_vertical_goucao2'  # 输出文件名(无后缀)
        #PLANE_PARAMS_NPY_FILE = 'planeParams_vertical_goucao2.npy'



    UPPER_SUBJECT_IMAGE_FILE = CALIB_DATA_PATH + '/Pic_20250517150814551.png'
    LOWER_SUBJECT_IMAGE_FILE = CALIB_DATA_PATH + '/Pic_20250517150623687.png'
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
    #               第二部分：物体三维轮廓重建及重投影
    # ==========================================================================
    print("\n" + "=" * 60)
    print("                第二部分: 物体表面三维重建")
    print("=" * 60)

    # if not os.path.exists(UPPER_SUBJECT_IMAGE_FILE):
    #     sys.exit(f"错误: 找不到目标物体图像 '{UPPER_SUBJECT_IMAGE_FILE}'，程序终止。")
    # if not os.path.exists(LOWER_SUBJECT_IMAGE_FILE):
    #     sys.exit(f"错误: 找不到目标物体图像 '{LOWER_SUBJECT_IMAGE_FILE}'，程序终止。")
    # # 检查平面参数文件是否已成功生成或原已存在
    # if not os.path.exists(PLANE_PARAMS_NPY_FILE):
    #     print(f"错误：激光平面参数文件 '{PLANE_PARAMS_NPY_FILE}' 未找到。")
    #     sys.exit("无法继续进行三维重建。请先确保第一部分的平面标定成功。")
    # print(f"已检测到激光平面参数文件: '{PLANE_PARAMS_NPY_FILE}'")
    # try:
    #     laser_plane_parameters = np.load(PLANE_PARAMS_NPY_FILE)
    #     print(f"成功加载激光平面参数: {laser_plane_parameters}")
    # except Exception as e:
    #     sys.exit(f"错误: 加载激光平面参数文件 '{PLANE_PARAMS_NPY_FILE}' 失败: {e}")

    # smoothed_upper_contour = reconstruction_and_projection(UPPER_SUBJECT_IMAGE_FILE, camera_parameters, laser_plane_parameters)
    # smoothed_lower_contour = reconstruction_and_projection(LOWER_SUBJECT_IMAGE_FILE, camera_parameters, laser_plane_parameters)
    # area, x_fill, y_upper_fill, y_lower_fill = calculate_area_between_contours(smoothed_upper_contour, smoothed_lower_contour)
    #
    # print("\n" + "=" * 70)
    # print(f"【最终结果】 计算出的横截面面积为: {-area:.4f} (平方毫米)")
    # print("=" * 70)

    # --- 最终结果可视化 --- 先不显示了
    # plt.figure(figsize=(12, 8))
    # ax = plt.gca()
    # # 绘制上轮廓线
    # ax.plot(smoothed_upper_contour[:, 0], smoothed_upper_contour[:, 1], 'b-', lw=2, label='上轮廓')
    # # 绘制下轮廓线
    # ax.plot(smoothed_lower_contour[:, 0], smoothed_lower_contour[:, 1], 'g-', lw=2, label='下轮廓')
    #
    # # 填充上下轮廓之间的区域
    # ax.fill_between(
    #     x_fill,  # 使用公共X轴
    #     y_upper_fill,  # 使用插值后的上Y值
    #     y_lower_fill,  # 使用插值后的下Y值
    #     color='skyblue',
    #     alpha=1,
    #     label=f'计算面积区域\n(Area = {area:.4f})'
    # )
    #
    # ax.set_title('物体横截面面积计算结果')
    # ax.set_xlabel('X (俯视图坐标)')
    # ax.set_ylabel('Y (俯视图坐标)')
    # ax.legend()
    # ax.grid(True)
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()

    CAMERA_PARAMS_FILE = CAMERA_PARAMS_MAT_FILE#MATLAB转好的相机参数
    LASER_PLANE_FILE = PLANE_PARAMS_NPY_FILE#命名激光平面
    #IMAGE_LOWER_CONTOUR_FIXED = '20250517/Pic_20250517150623687.png'  # <--- 【固定】的下轮廓图像
    ROI_CACHE_FILE = 'roi_cache.npy'  # 用于缓存上轮廓ROI的文件
    LOWER_CONTOUR_CACHE_FILE = 'lower_contour_cache.npy'  # <-- 缓存处理好的下轮廓
    # --- 加载公共参数 ---
    cam_params = load_matlab_camera_params(CAMERA_PARAMS_FILE)
    if not cam_params: sys.exit("相机参数加载失败。")

    try:
        plane_params = np.load(LASER_PLANE_FILE)
    except Exception as e:
        sys.exit(f"激光平面参数加载失败: {e}")

    # --- 【一次性准备】处理固定的下轮廓 ---
    #
    cached_lower_contour = None
    if os.path.exists(LOWER_CONTOUR_CACHE_FILE):
        cached_lower_contour = np.load(LOWER_CONTOUR_CACHE_FILE)
        print(f"成功从 '{LOWER_CONTOUR_CACHE_FILE}' 加载已处理的下轮廓数据。")
    else:
        print("\n" + "*" * 25 + " 一次性设置：处理下轮廓 " + "*" * 25)
        print("请为固定的【下轮廓】图像选择ROI。此操作仅需执行一次。")
        # lower_image_data= cv2.imread(IMAGE_LOWER_CONTOUR_FIXED)
        # lower_contour, _ = process_full_pipeline(lower_image_data, cam_params, plane_params)

        if lower_contour is not None:
            np.save(LOWER_CONTOUR_CACHE_FILE, lower_contour)
            print(f"下轮廓处理完毕，并已缓存到 '{LOWER_CONTOUR_CACHE_FILE}'")
            cached_lower_contour = lower_contour
        else:
            sys.exit("固定的下轮廓处理失败，无法继续。请检查图像或ROI选择。")

    # --- 正式调用功能 ---
    # 你可以在这里指定任何一张上轮廓图像

    # 调用核心函数
    calculated_area = get_area_from_upper_image(
        upper_image_data=UPPER_IMAGE_TO_TEST,
        fixed_lower_contour=cached_lower_contour,
        camera_params=cam_params,
        laser_plane_params=plane_params,
        ROI_CACHE_FILE=ROI_CACHE_FILE
    )

    if calculated_area is not None:
        print("\n流程执行完毕。")
        return calculated_area
    else:
        print("\n流程执行中出现错误。")

if __name__ == "__main__":
    UPPER_IMAGE_TO_TEST = cv2.imread('20250517/Pic_20250517150801704.png')
    van(UPPER_IMAGE_TO_TEST)