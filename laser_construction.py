import numpy as np
import scipy.io as sio
import open3d as o3d
import sys
import os


def reconstruct_laser_line(camera_params: dict, plane_params: np.ndarray, subject_uv: np.ndarray):
    """
    根据2D激光点、相机参数和激光平面方程，重建三维激光线。
    这是 MATLAB 函数 laser_construction 的 Python 实现。

    Args:
        camera_params (dict): 包含相机内参、外参的字典。
        plane_params (np.ndarray): 激光平面的方程参数 [a, b, c, d]。
        subject_uv (np.ndarray): 形状为 (N, 2) 的 numpy 数组，包含目标激光线的 [u,v] 像素坐标。

    Returns:
        np.ndarray: 成功时返回在相机坐标系下的三维点 (N, 3) numpy 数组，失败则返回 None。
    """

    # 1.从相机参数字典中获取内参矩阵
    intrinsic_matrix = camera_params['intrinsic_matrix']

    # 2. 过滤数据并提取 u, v 坐标 (假设 u=1 为无效点)
    u, v = subject_uv[:, 0], subject_uv[:, 1]
    valid_indices = np.where(u != 1)
    u_filtered, v_filtered = u[valid_indices], v[valid_indices]

    if len(u_filtered) == 0:
        print("错误：过滤后没有剩余的有效点。", file=sys.stderr)
        return None

    print(f"过滤掉 u=1 的点后，剩余 {len(u_filtered)} 个点用于重建。")

    # 3. 像素坐标转归一化图像坐标
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    u0, v0 = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    norm_x = (u_filtered - u0) / fx
    norm_y = (v_filtered - v0) / fy

    # 4. 解算相机坐标系下的三维点
    a, b, c, d = plane_params
    denominator = a * norm_x + b * norm_y + c

    # 检查分母是否接近零，防止除零错误
    if np.any(np.abs(denominator) < 1e-9):
        print("警告：计算中出现零分母，部分点可能无法重建。", file=sys.stderr)
        # 将接近零的分母替换为一个很小的数，以避免 NaN，后续可以根据需要过滤这些点
        denominator[np.abs(denominator) < 1e-9] = 1e-9

    z_camera = -d / denominator
    x_camera = norm_x * z_camera
    y_camera = norm_y * z_camera

    points_in_camera_coords = np.vstack((x_camera, y_camera, z_camera)).T
    print("已成功重建三维点（相机坐标系）。")

    # 5. [可选] 变换到世界坐标系并可视化
    R = camera_params['rotation_matrices'][0]
    T = camera_params['translation_vectors'][0].flatten()
    points_in_world_coords = (R.T @ (points_in_camera_coords - T).T).T
    print("已将三维点从相机坐标系转换到世界坐标系（用于可视化）。")

    # 6. 可视化三维点云 (使用 Open3D)
    print("正在使用 Open3D 可视化世界坐标系下的点云...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_in_world_coords)
    pcd.paint_uniform_color([0.0, 0.5, 1.0])

    # 添加一个坐标轴以更好地观察方向
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    #可视化 先不显示了
    # print("请在弹出的窗口中查看三维点云，关闭窗口后程序将继续。")
    # o3d.visualization.draw_geometries([pcd, coord_frame], window_name="3D Point Cloud (World Coordinates)")

    # 7. 保存并返回结果
    # output_camera_path = 'curve_data_InCamera.mat'
    # # MATLAB 中习惯将坐标矩阵保存为 (3, N)，所以需要转置
    # sio.savemat(output_camera_path, {'laser_camera': points_in_camera_coords.T})
    # print(f"成功将相机坐标点云保存到 '{output_camera_path}'")

    return points_in_camera_coords


# ==============================================================================
#                                主程序入口
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 配置文件路径 ---
    from load_camera_params import load_matlab_camera_params
    CAMERA_PARAMS_MAT_FILE = 'camera_params_for_python.mat'
    LASER_PLANE_PARAMS_FILE = 'planeParams_vertical_goucao3.npy'
    UV_DATA_FILE = 'uv.mat'

    print("=" * 60)
    print("      运行: 从2D激光点重建3D线 (reconstruct_laser v2.0)")
    print("=" * 60)

    # --- 2. 检查并加载所有输入文件 ---
    # 检查文件是否存在
    required_files = [CAMERA_PARAMS_MAT_FILE, LASER_PLANE_PARAMS_FILE, UV_DATA_FILE]
    for f in required_files:
        if not os.path.exists(f):
            sys.exit(f"错误: 输入文件缺失 '{f}'。请确保所有必需文件都在当前目录下。")

    # 加载相机参数
    camera_parameters = load_matlab_camera_params(CAMERA_PARAMS_MAT_FILE)
    if not camera_parameters:
        sys.exit("\n主程序终止：因相机参数加载失败，无法继续执行。")

    # 加载激光平面参数
    try:
        laser_plane_parameters = np.load(LASER_PLANE_PARAMS_FILE)
        print(f"成功从 '{LASER_PLANE_PARAMS_FILE}' 加载激光平面参数。")
    except Exception as e:
        sys.exit(f"错误: 加载激光平面参数文件 '{LASER_PLANE_PARAMS_FILE}' 失败: {e}")

    # 加载2D激光点 (uv坐标)
    try:
        subject_uv_data = sio.loadmat(UV_DATA_FILE)['uv']
        print(f"成功从 '{UV_DATA_FILE}' 加载了 {len(subject_uv_data)} 个2D激光点。")
    except Exception as e:
        sys.exit(f"错误: 加载2D激光点文件 '{UV_DATA_FILE}' 失败: {e}")

    # --- 3. 调用核心函数进行三维重建 ---
    reconstructed_points_camera = reconstruct_laser_line(
        camera_params=camera_parameters,
        plane_params=laser_plane_parameters,
        subject_uv=subject_uv_data  # <-- 直接传入加载好的 numpy 数组
    )

    # --- 4. 报告最终结果 ---
    if reconstructed_points_camera is not None:
        print("\n" + "-" * 60)
        print("主程序流程成功完成！")
        print(f"函数返回了 {len(reconstructed_points_camera)} 个点的三维坐标（相机系）。")
        print("返回的坐标数组形状为:", reconstructed_points_camera.shape)
    else:
        print("\n" + "-" * 60)
        print("主程序流程已结束，但未能成功重建三维点。")
