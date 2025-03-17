import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
import argparse

def read_labels(label_path, image_shape):
    """
    读取标注文件并解析为列表，同时验证坐标是否有效
    Args:
        label_path: 标注文件路径
        image_shape: 图像形状 (height, width)
    Returns:
        labels: 每行标注数据的列表
    """
    labels = []
    if not os.path.exists(label_path):
        print(f"警告: 标注文件 {label_path} 不存在")
        return labels
    height, width = image_shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split(' ')
            if len(data) == 9:  # 确保每行有类别和8个坐标值
                try:
                    class_id = data[0]
                    coords = list(map(float, data[1:]))
                    # 将相对坐标转换为绝对坐标
                    abs_coords = [
                        coords[i] * width if i % 2 == 0 else coords[i] * height
                        for i in range(len(coords))
                    ]
                    # 检查坐标是否在图像范围内
                    valid = all(0 <= abs_coords[i] < width and 0 <= abs_coords[i + 1] < height for i in range(0, 8, 2))
                    if not valid:
                        print(f"警告: {label_path} 中的坐标超出图像范围，跳过该行: {line}")
                        continue
                    labels.append([class_id] + abs_coords)
                except ValueError:
                    print(f"警告: {label_path} 中的数据格式错误，跳过该行: {line}")
    return labels

def write_labels(label_path, labels, image_shape):
    """
    将标签写入文件，将绝对坐标转换为相对坐标
    Args:
        label_path: 输出标注文件路径
        labels: 标签数据列表
        image_shape: 图像形状 (height, width)
    """
    height, width = image_shape[:2]
    with open(label_path, 'w') as f:
        for label in labels:
            class_id = label[0]
            coords = list(map(float, label[1:]))
            # 将绝对坐标转换为相对坐标
            rel_coords = [
                coords[i] / width if i % 2 == 0 else coords[i] / height
                for i in range(len(coords))
            ]
            f.write(f"{class_id} {' '.join(map(str, rel_coords))}\n")

def adjust_crop_region(points, image_shape):
    """
    根据四点标注调整裁剪区域，使裁剪区域与原图像保持相同比例，并增加扩展范围
    Args:
        points: 四点坐标 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        image_shape: 图像形状 (height, width)
    Returns:
        crop_region: 裁剪区域 [x_min, y_min, x_max, y_max]
        adjusted_points: 调整后的四点坐标
    """
    height, width = image_shape[:2]
    # 使用cv2.boundingRect来计算最小外接矩形
    points_np = np.array(points, dtype=np.float32)
    rect_x, rect_y, rect_w, rect_h = cv.boundingRect(points_np)
    
    # 增加扩展比例（例如扩展1.5倍）
    expand_ratio = 1.5
    new_width = int(rect_w * expand_ratio)
    new_height = int(rect_h * expand_ratio)
    
    # 计算外扩的区域
    margin_x = (new_width - rect_w) // 2
    margin_y = (new_height - rect_h) // 2
    
    # 调整裁剪区域
    rect_x = max(0, rect_x - margin_x)
    rect_y = max(0, rect_y - margin_y)
    rect_w = min(width, rect_x + new_width) - rect_x
    rect_h = min(height, rect_y + new_height) - rect_y
    
    # 调整四点坐标
    adjusted_points = [[p[0] - rect_x, p[1] - rect_y] for p in points]
    return [rect_x, rect_y, rect_x + rect_w, rect_y + rect_h], adjusted_points

def crop_and_resize(image, labels):
    """
    对图像进行裁剪并调整标签
    Args:
        image: 输入图像
        labels: 标签数据列表
    Returns:
        resized_image: 裁剪并调整大小后的图像
        adjusted_labels: 调整后的标签
    """
    height, width = image.shape[:2]
    adjusted_labels = []
    # 找到所有目标的四点坐标
    all_points = []
    for label in labels:
        coords = list(map(float, label[1:]))
        points = [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]
        all_points.extend(points)
    # 计算裁剪区域
    crop_region, _ = adjust_crop_region(all_points, image.shape)
    x_min, y_min, x_max, y_max = crop_region
    # 检查裁剪区域是否有效
    if x_min >= x_max or y_min >= y_max:
        print(f"警告: 裁剪区域无效，跳过该图像")
        return image, labels
    # 裁剪图像
    cropped_image = image[y_min:y_max, x_min:x_max]
    # Check if cropped image is empty (all black)
    if cropped_image.size == 0:
        print(f"警告: 裁剪区域无有效图像，跳过该图像")
        return image, labels
    # 获取裁剪后图像的尺寸
    cropped_height, cropped_width = cropped_image.shape[:2]
    # 调整标签坐标
    for label in labels:
        class_id = label[0]
        coords = list(map(float, label[1:]))
        points = [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]
        # 调整四点坐标
        adjusted_points = [[p[0] - x_min, p[1] - y_min] for p in points]
        # 将调整后的坐标映射到原始图像尺寸
        mapped_points = [
            [p[0] * width / cropped_width, p[1] * height / cropped_height] for p in adjusted_points
        ]
        # 转换为一维坐标
        mapped_coords = [coord for point in mapped_points for coord in point]
        adjusted_labels.append([class_id] + mapped_coords)
    # 将裁剪后的图像调整为原始大小
    resized_image = cv.resize(cropped_image, (width, height))
    return resized_image, adjusted_labels

def process_images(input_dir, output_dir):
    """
    处理目录中的所有图像和标签
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    # 获取图像和标签文件
    files = os.listdir(input_dir)
    image_files = [f for f in files if f.endswith(('.jpg', '.png'))]
    label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]
    # 创建进度条
    pbar = tqdm(total=len(image_files), desc="处理进度", ncols=100, dynamic_ncols=True)
    for img_file, label_file in zip(image_files, label_files):
        img_path = os.path.join(input_dir, img_file)
        label_path = os.path.join(input_dir, label_file)
        # 读取图像和标签
        image = cv.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}，跳过该文件")
            pbar.update(1)
            continue
        labels = read_labels(label_path, image.shape)
        if not labels:
            print(f"警告: {label_path} 中无有效标注，跳过 {img_file}")
            pbar.update(1)
            continue
        # 裁剪图像并调整标签
        resized_image, adjusted_labels = crop_and_resize(image, labels)
        # 保存结果
        output_img_path = os.path.join(output_dir, img_file)
        output_label_path = os.path.join(output_dir, label_file)
        cv.imwrite(output_img_path, resized_image)
        write_labels(output_label_path, adjusted_labels, image.shape)
        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({"当前文件": img_file})
    pbar.close()

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='图像裁剪与标签调整工具')
    parser.add_argument('--input', type=str, default='./input', help='输入目录路径')
    parser.add_argument('--output', type=str, default='./output', help='输出目录路径')
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    process_images(args.input, args.output)

if __name__ == "__main__":
    main()