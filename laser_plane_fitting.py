import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# # Matplotlib 默认不支持中文字符，需要额外设置
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
def _pixel_to_camera_coords(pixel_coord_homo: np.ndarray,
                            camera_matrix: np.ndarray,
                            rotation_matrix: np.ndarray,
                            translation_vector: np.ndarray) -> np.ndarray:
    """(内部辅助函数) 将单个像素点坐标转换为相机坐标系下的三维坐标。"""
    #求取Zc=1的相机系坐标
    inv_K = np.linalg.inv(camera_matrix)
    ray_direction_cam = inv_K @ pixel_coord_homo
    """原理：
    相机坐标系与世界坐标系有[x, y, z]' = R * [ X_w , Y_w , Z_w]' + T, (matlab中A' 表示 A的转置 )
    取标定板角点为世界坐标系原点，则标定板上左右点的 Z_w = 0
    进而 inv(R) * ([x, y, z]' -  T ) = [ X_w , Y_w , Z_w]'
    进而 inv(R)3 * ([x, y, z]' -  T ) = Z_w = 0   (取R逆的第三行对应的等式)
    即 r'31 * x + r'32 * y + r'33 * z = r'31 * t31 + r'32 * t32 + r'33 * t32
    上式右侧为常数，得到了一个平面方程，即标定板平面
    a * x + b * y + c * z = d  即为标定板平面, 可以通过R & T求得 a, b, c, d, 再根据归一化的x / z，y / z
    上式变化为[a * normlization(1) + b * normlization(2) + c] *z = d
    求得(x, y, z)
    """
    plane_normal_cam = np.linalg.inv(rotation_matrix)[2,:]
    plane_point_cam = translation_vector.flatten()
    s = (plane_normal_cam @ plane_point_cam) / (plane_normal_cam @ ray_direction_cam)
    point_in_cam_frame = s * ray_direction_cam
    return point_in_cam_frame


def calculate_laser_plane(
    laser_indices: list,
    point_counts: list,
    all_extracted_points: np.ndarray,
    camera_params: dict,
    output_filename: str,
    show_plot: bool = True) -> np.ndarray | None:

    """根据标定图像上的2D激光点，拟合出激光平面。
    Args:
        laser_indices (list): 一个整数列表，代表用于标定的图像在相机参数中的索引 (0-based)。
        point_counts (list): 一个整数列表，分别存储了从每张图像中提取的有效点的数量。
        all_extracted_points (np.ndarray): 一个 N x 3 的NumPy数组，包含了所有激光点的齐次坐标 [u, v, 1]。
        camera_params (dict): 一个包含相机标定参数的字典，由 `load_matlab_camera_params` 函数加载。
        output_filename (str, optional): 用于保存拟合平面参数的文件名。默认为 'plane_params_vertical.npy'。
        show_plot (bool, optional): 是否显示拟合结果的3D图。默认为 True。

    Returns:
        np.ndarray: 包含4个元素的激光平面方程参数 [a, b, c, d]，满足 a*x + b*y + c*z + d = 0。
    """
    cam_matrix = camera_params['intrinsic_matrix']
    rot_matrices = camera_params['rotation_matrices']
    trans_vectors = camera_params['translation_vectors']
    points_in_cam_frame = []
    cumulative_counts = np.cumsum([0] + point_counts)

    #二维图像坐标还原到世界坐标
    for i in range(all_extracted_points.shape[0]):
        pixel_coord = all_extracted_points[i]
        image_group_index = -1
        for k in range(len(point_counts)):
            if cumulative_counts[k] <= i < cumulative_counts[k+1]:
                image_group_index = k
                break
        if image_group_index == -1: continue
        actual_image_index = laser_indices[image_group_index]
        R = rot_matrices[actual_image_index]
        t = trans_vectors[actual_image_index]
        point_3d = _pixel_to_camera_coords(pixel_coord, cam_matrix, R, t)
        points_in_cam_frame.append(point_3d)

    loc = np.array(points_in_cam_frame)
    if loc.shape[0] < 3:
        print("错误：用于拟合的三维点少于3个，无法确定一个平面。")
        return None

    X, Y, Z = loc[:, 0], loc[:, 1], loc[:, 2]#X，Y，Z世界系坐标
    A = np.c_[X, Y]
    model = LinearRegression()
    model.fit(A, Z)
    p1, p2 = model.coef_
    p0 = model.intercept_
    plane_params_vertical = np.array([p1, p2, -1, p0])
    norm_factor = np.linalg.norm(plane_params_vertical[:3])
    plane_params_vertical /= norm_factor

    print("\n拟合的激光平面方程参数 [a, b, c, d]:")
    print(plane_params_vertical)

    if output_filename:
        # 确保输出文件名有.npy后缀
        if not output_filename.endswith('.npy'):
            output_filename += '.npy'
        np.save(output_filename, plane_params_vertical)
        print(f"平面参数已保存到 '{output_filename}'")

    if show_plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("激光平面拟合结果")
        ax.scatter(X, Y, Z, c='r', marker='o', label='激光三维点')
        x_fit = np.linspace(X.min(), X.max(), 10)
        y_fit = np.linspace(Y.min(), Y.max(), 10)
        X_FIT, Y_FIT = np.meshgrid(x_fit, y_fit)
        Z_FIT = -(plane_params_vertical[0] * X_FIT + plane_params_vertical[1] * Y_FIT + plane_params_vertical[3]) / plane_params_vertical[2]
        ax.plot_surface(X_FIT, Y_FIT, Z_FIT, alpha=0.5, color='cyan')
        ax.set_xlabel('X 轴 (毫米)'); ax.set_ylabel('Y 轴 (毫米)'); ax.set_zlabel('Z 轴 (毫米)')
        ax.legend(); ax.view_init(elev=20, azim=-60)
        plt.show()

    return plane_params_vertical
