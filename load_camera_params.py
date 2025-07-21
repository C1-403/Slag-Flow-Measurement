import numpy as np
import scipy.io as sio
def load_matlab_camera_params(mat_file_path: str) -> dict | None:
    """
    加载由MATLAB转换脚本生成的 .mat 文件。
    Args:
        mat_file_path (str): .mat文件的路径。

    Returns:
        dict | None: 一个包含相机参数的字典，如果加载成功。字典的键包括:
                     'intrinsic_matrix' (内参矩阵：列主序),
                     'rotation_matrices' (旋转矩阵列表：列主序),
                     'translation_vectors' (平移向量列表),
                     'dist_coeffs' (畸变系数[k1, k2, p1, p2, k3])。
                     如果加载失败，则返回 None。
    """
    try:
        print(f"正在从 '{mat_file_path}' 加载相机参数...")
        mat_data = sio.loadmat(mat_file_path)

        # 加载内参、旋转、平移矩阵
        # (MATLAB格式需要转置以符合通用标准)
        cam_matrix = mat_data['intrinsicMatrix'].T
        #【【非常重要】】MATLAB是行主序，所以rot_matrices = mat_data['rotationMatrices'].transpose(2, 0, 1)后还需要把矩阵转置一下，变成列主序
        rot_matrices = mat_data['rotationMatrices'].transpose(2, 1, 0)
        trans_vectors = mat_data['translationVectors']

        #  组合成OpenCV格式的畸变系数
        # 【新增】加载畸变系数
        # MATLAB的畸变系数通常是 [k1, k2, p1, p2, k3, ...]
        # 我们将径向畸变和切向畸变合并成一个数组
        radial_dist = mat_data['radialDistortion'].flatten()
        tangential_dist = mat_data['tangentialDistortion'].flatten()
        k1, k2 = radial_dist[0], radial_dist[1]
        p1, p2 = tangential_dist[0], tangential_dist[1]
        k3 = radial_dist[2] if len(radial_dist) > 2 else 0.0

        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

        print("相机参数加载成功（包含畸变系数）。")

        # 组装成一个字典并返回
        camera_params = {
            'intrinsic_matrix': cam_matrix.astype(np.float32),
            'rotation_matrices': rot_matrices.astype(np.float32),
            'translation_vectors': trans_vectors.astype(np.float32),
            'dist_coeffs': dist_coeffs.astype(np.float32)
        }
        return camera_params

    except FileNotFoundError:
        print(f"错误: 文件 '{mat_file_path}' 未找到。请确保文件路径正确。")
        return None
    except KeyError as e:
        print(f"错误: 在.mat文件中未找到必需的键: {e}。请确保MATLAB转换脚本已正确保存所有变量。")
        return None
    except Exception as e:
        print(f"加载.mat文件时发生未知错误: {e}")
        return None

