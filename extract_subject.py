# 文件名: extract_subject.py

import numpy as np
import cv2
import sys
from scipy.spatial.distance import cdist

# 从我们刚刚创建的文件中导入新的ROI选择函数
from matplotlib_roi_selector import select_roi_with_matplotlib


def extract_subject(image_data: str, camera_params: dict, roi_rect: tuple = None):
    """
    通过交互式或预设ROI和Steger算法从图像中提取激光中心线。
    【完美版本】:
    - 使用Matplotlib作为ROI选择器，解决了与cv2的GUI后端冲突。
    - 可接受一个预设的ROI矩形 `roi_rect`。
    - 返回提取到的点和实际使用的ROI。

    Args:
        image_path (str): 输入图像的路径。
        camera_params (dict): 相机参数字典。
        roi_rect (tuple, optional): 一个预设的ROI矩形 (x, y, w, h)。
                                    如果为None，则会弹出窗口让用户手动选择。

    Returns:
        tuple: 一个元组 `(filtered_points, used_roi)`:
               - np.ndarray: 成功时为 (N, 2) 的点集，失败时为 None。
               - tuple: 实际使用的ROI (x, y, w, h)，失败时为 None。
    """
    # 1. 读取图像并去畸变
    try:
        if image_data is None:
            raise ValueError("传入的图像数据为 None。")
        # 后续流程不变：去畸变、转灰度
        img_undistorted = cv2.undistort(image_data, camera_params['intrinsic_matrix'], camera_params['dist_coeffs'])
        gray_img = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"错误: 处理图像数据失败: {e}", file=sys.stderr)
        return None, None

    used_roi = None
    # 2. 获取ROI：使用预设的或调用新的Matplotlib选择器
    if roi_rect is not None and len(roi_rect) > 0:
        print(f"使用预设的ROI: {roi_rect}")
        roi = roi_rect
    else:
        roi = select_roi_with_matplotlib(gray_img)

    # 检查ROI是否有效
    if len(roi) == 0 or roi[2] == 0 or roi[3] == 0:
        print("错误: 未提供或未选择有效的ROI，程序终止。")
        return None, None
    used_roi = roi

    # 3. 裁剪图像并应用Steger算法
    x_roi, y_roi, w_roi, h_roi = [int(v) for v in roi]
    cropped_img = gray_img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

    # 注意：这里可以加入一个判断，如果裁剪区域为空则直接失败
    if cropped_img.size == 0:
        print(f"错误: 基于ROI {roi} 裁剪出的图像为空。", file=sys.stderr)
        return None, used_roi

    print("正在使用 Steger 算法提取中心线...")
    # Steger算法具体实现 (这部分逻辑保持不变)
    blurred = cv2.GaussianBlur(cropped_img, (13, 13), 3)
    dy, dx = np.gradient(blurred.astype(np.float64))
    _, dxx = np.gradient(dx)
    dyy, _ = np.gradient(dy)
    _, dxy = np.gradient(dy)
    h_crop, w_crop = blurred.shape
    uv_subpixel = []
    for r_idx in range(1, h_crop - 1):
        for c_idx in range(1, w_crop - 1):
            if blurred[r_idx, c_idx] < 10: continue
            H_mat = np.array([[dxx[r_idx, c_idx], dxy[r_idx, c_idx]], [dxy[r_idx, c_idx], dyy[r_idx, c_idx]]])
            eigenvalues, eigenvectors = np.linalg.eig(H_mat)
            max_eig_idx = np.argmax(np.abs(eigenvalues))
            nx, ny = eigenvectors[:, max_eig_idx]
            denom = dxx[r_idx, c_idx] * nx ** 2 + 2 * dxy[r_idx, c_idx] * nx * ny + dyy[r_idx, c_idx] * ny ** 2
            if abs(denom) < 1e-9: continue
            t = -(dx[r_idx, c_idx] * nx + dy[r_idx, c_idx] * ny) / denom
            if abs(t * nx) <= 0.5 and abs(t * ny) <= 0.5:
                # 坐标需要转换回原始图像坐标系
                uv_subpixel.append([c_idx + t * nx + x_roi, r_idx + t * ny + y_roi])

    if not uv_subpixel:
        print("错误: 在指定的ROI内未能提取到任何激光点。")
        return None, used_roi

    # 4. 使用KNN去噪
    uv_points = np.array(uv_subpixel)
    k, distance_threshold_factor = 5, 1.1
    num_points = uv_points.shape[0]
    if num_points <= k:
        filtered_points = uv_points
    else:
        # 使用cdist计算距离矩阵更高效
        distance_matrix = cdist(uv_points, uv_points)
        distance_matrix.sort(axis=1)
        avg_distances = np.mean(distance_matrix[:, 1:k + 1], axis=1)
        distance_threshold = np.mean(avg_distances) * distance_threshold_factor
        filtered_points = uv_points[np.where(avg_distances <= distance_threshold)[0]]

    print(f"成功提取并过滤了 {len(filtered_points)} 个激光点。")
    # 注意：此处不再需要用户按键确认结果，因为选择ROI的步骤已经完成。
    # 如果需要确认，可以重新加入 cv2.imshow 和 cv2.waitKey 的逻辑。
    # 但在一个自动化的流程中，通常选择省略这一步。

    return filtered_points, used_roi
