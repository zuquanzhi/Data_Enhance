import cv2
import numpy as np
import os
import argparse
import random

def create_output_folder(output_folder):
    """
    如果输出文件夹不存在，则创建输出文件夹
    :param output_folder: 输出文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")


def add_random_shadow(image):
    """
    向图像添加不规则阴影，阴影的位置、大小和深浅都是随机的
    :param image: 输入图像
    :return: 添加阴影后的图像
    """
    rows, cols, _ = image.shape
    x1 = random.randint(0, cols - 40)
    y1 = random.randint(0, rows - 40)
    x2 = random.randint(x1 + 20, cols)
    y2 = random.randint(y1 + 20, rows)
    
    # 随机选择阴影形状（椭圆）
    shadow = image.copy()
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (random.randint(30, 60), random.randint(30, 60))  # 随机轴长
    angle = random.randint(0, 180)
    
    # 随机深浅（透明度）
    alpha = random.uniform(0.2, 0.6)  # 阴影透明度
    
    # 生成阴影效果（椭圆形）
    cv2.ellipse(shadow, center, axes, angle, 0, 360, (0, 0, 0), -1)  # -1 填充阴影
    shadow = cv2.addWeighted(shadow, alpha, image, 1 - alpha, 0)  # 添加阴影透明效果
    
    return shadow


def add_random_noise(image):
    """
    向图像添加随机高斯噪声
    :param image: 输入图像
    :return: 添加噪声后的图像
    """
    row, col, ch = image.shape
    sigma = random.randint(15, 30)  # 随机噪声强度
    gauss = np.random.normal(0, sigma, (row, col, ch))
    noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy


def random_sharpen(image):
    """
    随机锐化图像（70%概率应用）
    :param image: 输入图像
    :return: 可能锐化后的图像
    """
    if random.random() < 0.7:  # 70%概率应用锐化
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    return image


def generate_yolo_label(image_path, output_txt_path):
    """
    生成YOLO格式标签文件，只保存原始图像坐标
    :param image_path: 输入标签文件路径
    :param output_txt_path: 输出标签文件路径
    :return: None
    """
    label_path = image_path.replace(os.path.splitext(image_path)[1], '.txt')  # 将图像文件的扩展名替换为 .txt
    
    all_lines = []
    
    try:
        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            print(f"标签文件不存在: {label_path}")
            return
        
        # 读取标签文件
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                if len(parts) < 9:  # 确保有类别和8个坐标值
                    print(f"忽略无效行: {line}")
                    continue
                
                all_lines.append(line)
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_txt_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 写入输出文件
        with open(output_txt_path, 'w', encoding='utf-8-sig') as f:
            f.writelines(all_lines)
            
    except Exception as e:
        print(f"处理标签文件时出错: {str(e)}")


def process_image(image, add_noise, add_shadow, random_sharp):
    """
    处理单张图像
    :return: 处理后的图像
    """
    if add_shadow:
        image = add_random_shadow(image)
    if add_noise:
        image = add_random_noise(image)
    if random_sharp:
        image = random_sharpen(image)
    return image


def process_single_image(input_path, output_folder, add_noise, add_shadow, random_sharp):
    create_output_folder(output_folder)

    # 处理单个图像
    filename = os.path.basename(input_path)
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        print(f"跳过非图像文件: {filename}")
        return

    # 读取图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"跳过无法读取的文件: {filename}")
        return

    # 处理图像
    processed_img = process_image(image, add_noise, add_shadow, random_sharp)
    
    # 保存处理后的图像
    output_img_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_img_path, processed_img)
    
    # 生成YOLO标签文件
    base_name = os.path.splitext(filename)[0]
    output_txt_path = os.path.join(output_folder, f"{base_name}.txt")
    generate_yolo_label(input_path, output_txt_path)

    print(f"图像处理完成！结果保存在: {output_folder}")


def process_images_in_folder(input_path, output_folder, add_noise, add_shadow, random_sharp):
    """
    如果输入路径是文件夹，处理文件夹中的所有图像文件。
    :param input_path: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            process_single_image(file_path, output_folder, add_noise, add_shadow, random_sharp)


def parse_arguments():
    parser = argparse.ArgumentParser(description="图像增强工具")
    parser.add_argument('input', help="输入图片路径或文件夹路径")
    parser.add_argument('output', help="输出文件夹路径")
    parser.add_argument('--noise', action='store_true', help="添加随机噪声")
    parser.add_argument('--shadow', action='store_true', help="添加随机阴影")
    parser.add_argument('--sharpen', action='store_true', help="随机锐化图像")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_path = args.input
    output_folder = args.output
    
    if os.path.isdir(input_path):
        process_images_in_folder(input_path, output_folder, args.noise, args.shadow, args.sharpen)
    elif os.path.isfile(input_path):
        process_single_image(input_path, output_folder, args.noise, args.shadow, args.sharpen)
    else:
        print("错误：无效的输入路径")