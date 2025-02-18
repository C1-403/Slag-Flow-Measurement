import cv2
import os


def extract_frames_per_second(video_path, output_dir, fps=1):
    # 创建输出目录（如果不存在）  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开  
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 初始化帧计数器
    frame_count = 0
    prev_time = 0

    # 逐帧读取视频
    while True:
        ret, frame = cap.read()

        # 如果读取帧失败（例如，视频结束），则退出循环
        if not ret:
            break

            # 获取当前帧的时间戳（以秒为单位）
        current_time = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        # 如果当前时间戳与上一次保存帧的时间戳之差大于或等于指定的fps，则保存当前帧
        if current_time - prev_time >= fps:
            # 构造输出文件名
            output_file = os.path.join(output_dir, f"image_{frame_count:04d}.png")

            # 保存当前帧为图片
            cv2.imwrite(output_file, frame)

            # 更新帧计数器和上一次保存帧的时间戳
            frame_count += 1
            prev_time = current_time


    # # 读取一帧
    # ret, frame = cap.read()
    #
    # # 如果成功读取了一帧，保存它
    # if ret:
    #     cv2.imwrite('first_frame.jpg', frame)

    # 释放VideoCapture对象
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 使用示例
    video_path = '../../new_videos/24.avi'  # 替换为你的视频文件路径
    output_dir = '../../new_videos/dataset/test'  # 替换为你希望保存帧图片的目录路径
    extract_frames_per_second(video_path, output_dir, fps=0.001)
