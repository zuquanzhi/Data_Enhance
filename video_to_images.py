import cv2
import os
from pathlib import Path
from tqdm import tqdm

def video_to_shot(video_path, output_path):
    """
    将视频转换为连续帧图像
    
    Args:
        video_path (str): 输入视频文件路径
        output_path (str): 输出图像保存目录
    
    Returns:
        int: 提取的帧数量，失败返回0
    """
    # 检查输入视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件 '{video_path}' 不存在")
        return 0

    # 创建输出目录
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'")
        return 0

    try:
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"视频信息：总帧数 {total_frames}, FPS {fps}")

        # 使用tqdm创建进度条
        pbar = tqdm(total=total_frames, desc="处理进度")
        frame_count = 0

        # 循环读取视频的每一帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 构建输出文件路径
            frame_path = output_dir / f"A_{frame_count:d}.jpg"
            
            # 保存当前帧为图片
            cv2.imwrite(str(frame_path), frame)

            # 更新进度条
            pbar.update(1)
            frame_count += 1

        pbar.close()
        print(f"完成！共保存 {frame_count} 帧图像到 '{output_path}'")
        return frame_count

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return 0

    finally:
        cap.release()

def main():
    """主函数"""
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='视频帧提取工具')
    parser.add_argument('--video', '-v', required=True,help='输入视频文件路径')
    parser.add_argument('--output', '-o', default='frames',help='输出目录路径 (默认: frames)')
    
    args = parser.parse_args()
    
    # 执行视频转换
    frames = video_to_shot(args.video, args.output)
    if frames > 0:
        print(f"成功提取 {frames} 帧图像")
    else:
        print("视频处理失败")

if __name__ == '__main__':
    main()