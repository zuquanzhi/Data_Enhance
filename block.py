import cv2
import numpy as np
import argparse

def parse_label(label_line):
    """解析标签行，返回类别ID和四个点的坐标
    Args:
        label_line: 标签文件中的一行字符串
    Returns:
        class_id: 类别ID
        points: 四个点的坐标数组
    """
    parts = list(map(float, label_line.strip().split()))
    class_id = int(parts[0])
    points = np.array(parts[1:]).reshape(4, 2)  # 转换为4个点的坐标
    return class_id, points

def draw_polygon_mask(image_shape, points):
    """绘制多边形的掩码
    Args:
        image_shape: 图像的形状 (height, width)
        points: 四个点的坐标
    Returns:
        mask: 多边形掩码
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = points.astype(np.int32)  # 确保点是整数类型
    cv2.fillPoly(mask, [points], 255)  # 填充多边形
    return mask

def generate_edge_mask(points, edge_index, image_shape, thickness=5):
    """生成某一条边的掩码
    Args:
        points: 四个点的坐标
        edge_index: 要隐藏的边的索引 (0, 1, 2, 3)
        image_shape: 输入图像的形状 (height, width)
        thickness: 边的厚度
    Returns:
        mask: 边缘掩码
    """
    # 创建与输入图像相同大小的掩码
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # 获取当前边的两个端点
    p1 = points[edge_index]
    p2 = points[(edge_index + 1) % 4]  # 下一个点
    
    # 绘制线段
    cv2.line(mask, tuple(p1.astype(int)), tuple(p2.astype(int)), 255, thickness=thickness)
    return mask

def inpaint_edge(image, mask):
    """使用OpenCV的inpaint方法修补掩码区域
    Args:
        image: 输入图像
        mask: 掩码
    Returns:
        inpainted_image: 修补后的图像
    """
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

def process_image(image_path, label_path, output_path, mode='random',thickness=5):
    """处理图像，隐藏指定边
    Args:
        image_path: 输入图像路径
        label_path: 标签文件路径
        output_path: 输出图像路径
        edge_index: 要隐藏的边的索引 (0, 1, 2, 3)
        thickness: 边的厚度
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请检查路径是否正确。")

    # 读取标签文件
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # 根据模式选择要处理的边
    if mode == 'left':
        edge_index = 2
    elif mode == 'right':
        edge_index = 0
    else:  # random
        edge_index = np.random.choice([0, 2])


    # 处理每个标注对象
    for label_line in labels:
        # 解析标签
        class_id, points = parse_label(label_line)

        # 将归一化坐标转换为实际像素坐标
        height, width = image.shape[:2]
        points[:, 0] *= width
        points[:, 1] *= height

        # 生成边的掩码，传入图像形状
        edge_mask = generate_edge_mask(points, edge_index, image.shape, thickness)

        # 修补图像
        image = inpaint_edge(image, edge_mask)

    # 保存结果
    cv2.imwrite(output_path, image)
    print(f"处理完成，结果已保存到 {output_path}")


import os
from tqdm import tqdm

def process_directory(input_dir, output_dir, mode='random', thickness=5):
    """处理整个目录的图像和标签文件
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        mode: 处理模式 ('left', 'right', 'random')
        thickness: 边的厚度
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    label_files = []
    for file in os.listdir(input_dir):
        name, ext = os.path.splitext(file)
        if ext.lower() in ['.jpg', '.jpeg', '.png']:
            image_files.append(file)
            label_file = f"{name}.txt"
            if os.path.exists(os.path.join(input_dir, label_file)):
                label_files.append(label_file)
    
    # 创建进度条
    pbar = tqdm(total=len(image_files), desc="处理进度")
    
    # 处理每个图像
    for image_file in image_files:
        name, _ = os.path.splitext(image_file)
        label_file = f"{name}.txt"
        
        if label_file not in label_files:
            pbar.update(1)
            continue
            
        try:
            # 生成输入输出路径
            image_path = os.path.join(input_dir, image_file)
            label_path = os.path.join(input_dir, label_file)
            output_image = os.path.join(output_dir, image_file)
            output_label = os.path.join(output_dir, label_file)
            
            # 处理图像
            process_image(image_path, label_path, output_image, mode, thickness)
            
            # 复制标签文件
            with open(label_path, 'r') as src:
                with open(output_label, 'w') as dst:
                    dst.write(src.read())
                    
        except Exception as e:
            print(f"处理 {image_file} 失败: {str(e)}")
            
        pbar.update(1)
        pbar.set_postfix({"当前文件": image_file})
    
    pbar.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="隐藏标注框的指定边")
    parser.add_argument("--input", type=str, required=True,
                      help="输入目录路径，包含图像和标注文件")
    parser.add_argument("--output", type=str, required=True,
                      help="输出目录路径")
    parser.add_argument("--mode", type=str, choices=['left', 'right', 'random'],
                      default='random', help="处理模式：left(左边), right(右边), random(随机)")
    parser.add_argument("--thickness", type=int, default=5,
                      help="边的厚度")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 确保输入目录存在
        if not os.path.exists(args.input):
            raise ValueError(f"输入目录不存在: {args.input}")
            
        # 处理整个目录
        process_directory(
            args.input,
            args.output,
            mode=args.mode,
            thickness=args.thickness
        )
        
        print(f"处理完成！输出目录: {args.output}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()