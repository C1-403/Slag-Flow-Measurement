import cv2
import numpy as np
from load_camera_params import load_matlab_camera_params

def extract_laser_points(image_path: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, threshold: int = 80) -> np.ndarray:
    """
    在激光标定图像上选取标定点，返回图像点的坐标。

    Args:
        image_path (str): 标定图片的路径。
        camera_matrix (np.ndarray): 相机内参矩阵 (3x3 NumPy array)。
        dist_coeffs (np.ndarray): 相机畸变系数 (1xN NumPy array)。
        threshold (int, optional): 用于过滤背景噪声的亮度阈值。默认为 80。

    Returns:
        np.ndarray: 一个 N x 3 的 NumPy 数组，包含了所有提取到的激光点的齐次坐标 [x, y, 1]。
                    如果用户取消操作，则返回 None。
    """
    # 1. 读取图像并进行去畸变
    try:
        img_original = cv2.imread(image_path)
        if img_original is None:
            print(f"错误：无法读取图像文件 {image_path}")
            return None
    except Exception as e:
        print(f"读取图像时发生错误: {e}")
        return None

    img = cv2.undistort(img_original, camera_matrix, dist_coeffs)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==================== BUG 修复关键代码 ====================
    # 1. 使用一个简单的、纯英文的窗口名
    window_name = "Select ROI"

    # 2. 在循环外显式地创建窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 1024)  # 可选：设置一个合适的窗口大小
    # ========================================================

    extracted_points = None

    while True:
        # 在选择ROI之前，先在窗口中显示图像，让用户知道要操作的图片
        cv2.imshow(window_name, gray_img)
        print("\n请在弹出的窗口中用鼠标拖拽一个矩形框来选择激光条纹区域。")
        print("选择完毕后，按 Enter 或 Space 键确认。")

        # 3. 让 selectROI 在我们已经创建好的窗口上工作
        roi = cv2.selectROI(window_name, gray_img, fromCenter=False, showCrosshair=True)

        # selectROI 会“暂停”代码，直到用户确认。确认后，它会返回ROI
        # 并且窗口会保持显示状态，我们需要手动管理它。

        if roi[2] == 0 or roi[3] == 0:
            print("未选择有效区域，操作取消。")
            cv2.destroyWindow(window_name)  # 只关闭这一个窗口
            return None

        x, y, w, h = roi

        # 在ROI内提取激光中心点 (这部分逻辑不变)
        laser_points = []
        center_line_img = np.zeros_like(gray_img)
        for j in range(x, x + w):
            roi_column = gray_img[y:y + h, j]
            max_val = np.max(roi_column)
            if max_val > threshold:
                local_max_idx = np.argmax(roi_column)
                global_i = y + local_max_idx
                laser_points.append([j, global_i])
                center_line_img[global_i, j] = 255

        # 显示结果并请求用户确认
        preview_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for point in laser_points:
            cv2.circle(preview_img, tuple(point), 1, (0, 0, 255), -1)

        cv2.imshow("Extracted Laser Centerline", center_line_img[y:y + h, x:x + w])
        cv2.imshow("Result Preview", preview_img)

        print("\n结果是否满意？")
        print("  - 按 'y' 或 'Y' 键确认并保存。")
        print("  - 按 'n' 或 'N' 键重新选择区域。")

        key = cv2.waitKey(0) & 0xFF

        # 关闭用于确认结果的预览窗口
        cv2.destroyWindow("Extracted Laser Centerline")
        cv2.destroyWindow("Result Preview")

        if key == ord('y'):
            points_xy = np.array(laser_points, dtype=np.float32)
            ones_column = np.ones((points_xy.shape[0], 1), dtype=np.float32)
            extracted_points = np.hstack([points_xy, ones_column])
            print(f"操作成功，已提取 {len(extracted_points)} 个点。")
            break  # 结束循环
        elif key == ord('n'):
            print("用户选择“否”，请重新绘制矩形框。")
            # 循环会继续，回到顶部再次调用 cv2.imshow 和 cv2.selectROI
        else:
            print("无效按键，操作将取消。")
            break

    # 在函数结束时，确保所有OpenCV窗口都被关闭
    cv2.destroyAllWindows()
    return extracted_points


# --- 如何使用这个函数的示例 ---
if __name__ == '__main__':
    #导入相机参数、图像路径
    converted_mat_file = 'D:\MATLAB\laser-triangulation-master-smooth\camera_params_for_python.mat'
    camera_matrix,_,_, dist_coeffs = load_matlab_camera_params(converted_mat_file)
    if camera_matrix is not None and dist_coeffs is not None:
        print("\n成功从 .mat 文件加载相机参数:")
        print("Camera Matrix (K):\n", camera_matrix)
        print("Distortion Coefficients [k1,k2,p1,p2,k3]:\n", dist_coeffs)
        print("-" * 50)
    image_path = '20250523/1.png'

    # 调用函数
    points = extract_laser_points(image_path, camera_matrix, dist_coeffs)

    # 4. 打印结果
    if points is not None:
        print("\n提取到的激光点坐标 (前10个):")
        print("格式: [x, y, 1]")
        print(points[:10])
        print(f"\n总共提取到 {points.shape[0]} 个点。")
    else:
        print("\n没有提取到任何点或操作被用户取消。")
