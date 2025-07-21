import ast
import os
import sys
import numpy as np
from load_camera_params import load_matlab_camera_params
from extract_laser import extract_laser_points
from laser_plane_fitting import calculate_laser_plane

# ==========================================================================
# 本文件在所有流程前单独执行，调用calibaration_and_save_laser_plane（）进行激光平面标定
# ==========================================================================


def load_config_from_file(filepath: str) -> dict:
    """
    从一个文本文件中读取配置参数。

    Args:
        filepath: 配置文件的路径。

    Returns:
        一个包含所有配置参数的字典。
    """
    config = {}
    print(f"--- 正在从 '{filepath}' 读取配置 ---")

    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"错误：配置文件 '{filepath}' 未找到。")
        return config

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. 清理行首和行尾的空白字符
            line = line.strip()

            # 2. 跳过空行或注释行 (以 # 开头)
            if not line or line.startswith('#'):
                continue

            # 3. 按第一个 '=' 分割成键和值
            parts = line.split('=', 1)
            if len(parts) != 2:
                # 跳过格式不正确的行
                continue

            key = parts[0].strip()
            value_str = parts[1].strip()

            # 4. 使用 ast.literal_eval 安全地将字符串转换为Python对象
            # 这是最关键的一步，它可以正确处理字符串、列表、数字等
            try:
                value = ast.literal_eval(value_str)
                config[key] = value
                print(f"成功加载: {key} = {value} (类型: {type(value).__name__})")
            except (ValueError, SyntaxError):
                # 如果转换失败 (例如，值不是一个有效的Python字面量),
                # 将其作为普通字符串处理。
                config[key] = value_str
                print(f"警告：无法解析 '{value_str}'，已将其作为普通字符串加载。")

    print("--- 配置加载完毕 ---\n")
    return config
def calibaration_and_save_laser_plane():
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



    camera_parameters = load_matlab_camera_params(CAMERA_PARAMS_MAT_FILE)
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

# --- 主程序入口 ---
if __name__ == "__main__":
    calibaration_and_save_laser_plane()
