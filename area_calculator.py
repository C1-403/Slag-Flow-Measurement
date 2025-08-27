# ==============================================================================
#           area_calculator.py (v1.0)
#
#   主功能:
#   - 首次运行时，处理并缓存固定的下轮廓。
#   - 提供一个函数，输入上轮廓图像，自动计算横截面面积。
#   - 自动缓存和重用ROI选择，实现自动化处理。
# ==============================================================================

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from load_camera_params import load_matlab_camera_params
from extract_subject import extract_subject
from laser_construction import reconstruct_laser_line
from project_and_smooth import project_and_smooth
from reconstruction_and_projection import calculate_area_between_contours
from configuration_and_calbration import load_config_from_file

# --- 全局设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def process_full_pipeline(image_data: np.ndarray, camera_params: dict, laser_plane_params: np.ndarray, roi_rect=None, SAVE_ROI=False):

    # 1. 调用新版的 extract_subject
    uv_points, used_roi = extract_subject(image_data, camera_params, roi_rect=roi_rect)
    if uv_points is None:
        # 日志信息不再包含路径
        print(f"未能从提供的图像数据中提取激光点。")
        return None, None

    # 2.点云重建
    points_3d = reconstruct_laser_line(camera_params, laser_plane_params, uv_points)
    if points_3d is None:
        print(f"未能从激光点重建3D坐标。")
        return None, used_roi
    # 3.重投影
    smoothed_contour = project_and_smooth(laser_plane_params, points_3d, show_plot=False)
    if smoothed_contour is None:
        print(f"未能对3D点进行投影平滑。")
        return None, used_roi
    if SAVE_ROI and used_roi is not None:
        ROI_CACHE_FILE='roi_cache.npy'
        np.save(ROI_CACHE_FILE, used_roi)
        print(f"新的ROI已成功选择并保存到 '{ROI_CACHE_FILE}'")

    return smoothed_contour, used_roi

def save_lower_countour(image_data: np.ndarray,SAVE_ROI=False):
    LOWER_CONTOUR_CACHE_FILE = 'lower_contour_cache.npy'  # 设置即将保存的下轮廓文件名
    config_data = load_config_from_file('config.txt')
    # 检查配置是否加载成功
    if not config_data:
        print("脚本因配置加载失败而退出。")
    else:
        CAMERA_PARAMS_MAT_FILE = config_data.get('CAMERA_PARAMS_MAT_FILE')
        PLANE_PARAMS_OUTPUT_FILE = config_data.get('PLANE_PARAMS_OUTPUT_FILE')
        PLANE_PARAMS_NPY_FILE = PLANE_PARAMS_OUTPUT_FILE + '.npy'
    camera_parameters = load_matlab_camera_params(CAMERA_PARAMS_MAT_FILE)
    try:
        plane_params = np.load(PLANE_PARAMS_NPY_FILE)
    except Exception as e:
        print(f"激光平面参数加载失败: {e}")
    #调用核心提取-重建-重投影函数
    lower_contour, _ = process_full_pipeline(image_data, camera_parameters,plane_params,roi_rect=None,SAVE_ROI=SAVE_ROI)
    if lower_contour is not None:
        np.save(LOWER_CONTOUR_CACHE_FILE, lower_contour)
        print(f"下轮廓处理完毕，并已缓存到 '{LOWER_CONTOUR_CACHE_FILE}'")
        return lower_contour
    else:
        print("固定的下轮廓处理失败，无法继续。请检查图像或ROI选择。")
        return None



def get_area_from_upper_image(upper_image_data: np.ndarray, fixed_lower_contour: np.ndarray, camera_params: dict,
                              laser_plane_params: np.ndarray, ROI_CACHE_FILE):
    """
    核心功能函数：输入上轮廓图，计算与固定下轮廓的面积。
    自动处理ROI缓存。
    """
    print("\n" + "=" * 30 + f" 开始处理上轮廓图像: " + "=" * 30)

    # 1. 确定上轮廓的ROI
    roi_to_use = None
    if os.path.exists(ROI_CACHE_FILE):
        roi_to_use = np.load(ROI_CACHE_FILE)
        print(f"成功加载缓存的ROI: {roi_to_use}")
    else:
        print("未找到缓存的ROI文件，将需要手动选择。")

    # 2. 处理上轮廓
    smoothed_upper, used_roi = process_full_pipeline(upper_image_data, camera_params, laser_plane_params,
                                                     roi_rect=roi_to_use)
    if smoothed_upper is None:
        return None  # 如果处理失败，返回None

    # 3. 如果是首次运行（没有用缓存），则保存新选择的ROI
    if not os.path.exists(ROI_CACHE_FILE) and used_roi is not None:
        np.save(ROI_CACHE_FILE, used_roi)
        print(f"新的ROI已成功选择并缓存到 '{ROI_CACHE_FILE}'")

    # 4. 计算面积
    area, x_fill, y_upper_fill, y_lower_fill = calculate_area_between_contours(smoothed_upper, fixed_lower_contour)
    area *= -1
    area /= 10000
    print("\n" + "=" * 70)
    print(f"【计算结果】 截面面积: {-1 * area:.4f} (m^2)")
    print("=" * 70)

    # 5. 可视化
    # plt.figure(figsize=(12, 8))
    # ax = plt.gca()
    # ax.plot(smoothed_upper[:, 0], smoothed_upper[:, 1], 'b-', lw=2, label='上轮廓')
    # ax.plot(fixed_lower_contour[:, 0], fixed_lower_contour[:, 1], 'g-', lw=2, label='下轮廓 (固定)')
    # ax.fill_between(x_fill, y_upper_fill, y_lower_fill, color='skyblue', alpha=0.5,
    #                 label=f'截面面积\n(Area = {area:.4f} mm²)')
    # ax.set_title(f'横截面面积分析: ')
    # ax.set_xlabel('X 坐标 (mm)')
    # ax.set_ylabel('Y 坐标 (mm)')
    # ax.legend()
    # ax.grid(True)
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()

    return area


if __name__ == '__main__':
    import cv2
    # --- 1. 全局配置区域 ---
    # !! 请务必根据您的文件修改这里的路径 !!
    CAMERA_PARAMS_FILE = 'camera_params_for_python.mat'
    LASER_PLANE_FILE = 'planeParams_vertical_goucao2.npy'
    ROI_CACHE_FILE = 'roi_cache.npy'  # <--- 用于缓存上轮廓ROI的文件
    LOWER_CONTOUR_CACHE_FILE = 'lower_contour_cache.npy'  # <-- 缓存处理好的下轮廓
    # --- 加载公共参数 ---
    cam_params = load_matlab_camera_params(CAMERA_PARAMS_FILE)
    if not cam_params: sys.exit("相机参数加载失败。")

    try:
        plane_params = np.load(LASER_PLANE_FILE)
    except Exception as e:
        sys.exit(f"激光平面参数加载失败: {e}")

    # --- 【一次性准备】处理固定的下轮廓 ---
    '''cached_lower_contour = None
    if os.path.exists(LOWER_CONTOUR_CACHE_FILE):
        cached_lower_contour = np.load(LOWER_CONTOUR_CACHE_FILE)
        print(f"成功从 '{LOWER_CONTOUR_CACHE_FILE}' 加载已处理的下轮廓数据。")
    else:
        print("\n" + "*" * 25 + " 一次性设置：处理下轮廓 " + "*" * 25)
        print("请为固定的【下轮廓】图像选择ROI。此操作仅需执行一次。")
        #调用下轮廓提取函数，覆盖ROI和下轮廓文件
        IMAGE_LOWER_CONTOUR_FIXED = cv2.imread('20250517/Pic_20250517150623687.png')  # <--- 【固定】的下轮廓图像
        get_lower_countour(IMAGE_LOWER_CONTOUR_FIXED)'''
    #下面2行用于测试能否保存下轮廓函数，和上面'''  '''代码互相替换
    # 调用下轮廓提取函数，覆盖ROI和下轮廓文件
    image_data = cv2.imread('20250517/Pic_20250517150623687.png')  # <--- 【固定】的下轮廓图像
    cached_lower_contour=save_lower_countour(image_data,SAVE_ROI=True)

    # --- 正式调用功能 ---
    # 你可以在这里指定任何一张上轮廓图像
    UPPER_IMAGE_TO_TEST = cv2.imread('20250517/Pic_20250517150814551.png')

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
    else:
        print("\n流程执行中出现错误。")
