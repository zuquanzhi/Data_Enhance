import cv2
import numpy as np
import os
import argparse

def create_output_path(output_path):
    """
    如果输出文件夹不存在，则创建输出文件夹
    :param output_path: 输出文件夹路径
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Output folder created: {output_path}")

def read_labels(txt_path, output_path, img_width, img_height):
    """
    读取标签文件并返回坐标（将归一化坐标转换为绝对坐标）
    :param txt_path: 标签文件路径
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: 坐标列表，每个坐标点包含四个点组成的多边形
    """
    labels = []
    all_lines = []  # 存储所有要写入的行
    
    if not os.path.exists(txt_path):
        print(f"Label file {txt_path} does not exist.")
        return labels
    
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            if len(parts) < 9:  # 确保有类别和8个坐标值
                print(f"Ignore invalid line in {txt_path}: {line}")
                continue
                
            coords = [float(x) for x in parts[1:]]  # 跳过类别
            all_lines.append(line)  # 保存原始行内容
            
            if len(coords) % 2 != 0:
                print(f"Ignore invalid coordinates in {txt_path}: {line}")
                continue
                
            # 转换为绝对坐标
            abs_coords = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_width
                y = coords[i+1] * img_height
                abs_coords.append([x, y])
            coords_array = np.array(abs_coords, dtype=np.float32)
            labels.append(coords_array)

    # 在循环结束后统一写入标签文件
    if output_path:
        base_name = os.path.basename(txt_path)
        output_label_path = os.path.join(output_path, base_name)
        with open(output_label_path, 'w') as f:
            f.writelines(all_lines)  # 一次性写入所有行
    
    return labels
            


def process_image(image, alpha=1.6, beta=25, mode="full_image", labels=None):
    """
    处理单张图像，增加对比度和亮度，并返回处理后的图像
    :param image: 输入图像
    :param alpha: 对比度因子，默认为1.6
    :param beta: 亮度因子，默认为25
    :param mode: 处理模式
    :param labels: 标签区域的坐标
    :return: 处理后的图像
    """
    if mode in ["inside_polygon", "outside_polygon"] and (not labels):
        print("Warning: No labels provided for polygon mode. Returning original image.")
        return image.copy()
    
    
    if mode == "random":
        random_alpha = np.random.uniform(1.0, alpha)  # alpha范围：1.0到指定值
        random_beta = np.random.randint(0, beta)      # beta范围：0到指定值
        return cv2.convertScaleAbs(image, alpha=random_alpha, beta=random_beta)

    
    if mode == "full_image":
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for label in labels:
        label_int = np.round(label).astype(np.int32)
        cv2.fillPoly(mask, [label_int], 255)
    
    if mode == "inside_polygon":
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        adjusted_masked = cv2.bitwise_and(adjusted, adjusted, mask=mask)
        orig_masked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        return cv2.add(orig_masked, adjusted_masked)
    
    elif mode == "outside_polygon":
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        outside_mask = cv2.bitwise_not(mask)
        adjusted_masked = cv2.bitwise_and(adjusted, adjusted, mask=outside_mask)
        orig_masked = cv2.bitwise_and(image, image, mask=mask)
        return cv2.add(orig_masked, adjusted_masked)
    
    # elif mode == "superrandom":
    #     # 对每个多边形区域使用不同的随机参数
    #     result = image.copy()
    #     for label in labels:
    #         # 为每个区域生成独立的随机参数
    #         random_alpha = np.random.uniform(1.0, alpha)  # 随机对比度
    #         random_beta = np.random.randint(0, beta)      # 随机亮度
            
    #         # 创建单个区域的掩码
    #         single_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    #         label_int = np.round(label).astype(np.int32)
    #         cv2.fillPoly(single_mask, [label_int], 255)
            
    #         # 处理当前区域
    #         adjusted = cv2.convertScaleAbs(image, alpha=random_alpha, beta=random_beta)
    #         adjusted_masked = cv2.bitwise_and(adjusted, adjusted, mask=single_mask)
    #         orig_masked = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(single_mask))
    #         result = cv2.add(orig_masked, adjusted_masked)

    elif mode == "superrandom":
        # 对每个区域（包括多边形内部和外部）使用不同的随机参数
        result = image.copy()
        
        # 首先处理所有多边形外部区域
        outside_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255  # 创建全白掩码
        for label in labels:
            label_int = np.round(label).astype(np.int32)
            cv2.fillPoly(outside_mask, [label_int], 0)  # 将多边形区域置为黑色
        
        # 处理外部区域
        random_alpha = np.random.uniform(1.0, alpha)
        random_beta = np.random.randint(0, beta)
        adjusted = cv2.convertScaleAbs(image, alpha=random_alpha, beta=random_beta)
        adjusted_masked = cv2.bitwise_and(adjusted, adjusted, mask=outside_mask)
        result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(outside_mask))
        result = cv2.add(result, adjusted_masked)
        
        # 然后处理每个多边形内部区域
        for label in labels:
            # 为每个区域生成独立的随机参数
            random_alpha = np.random.uniform(1.0, alpha)
            random_beta = np.random.randint(0, beta)
            
            # 创建单个区域的掩码
            single_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            label_int = np.round(label).astype(np.int32)
            cv2.fillPoly(single_mask, [label_int], 255)
            
            # 处理当前区域
            adjusted = cv2.convertScaleAbs(image, alpha=random_alpha, beta=random_beta)
            adjusted_masked = cv2.bitwise_and(adjusted, adjusted, mask=single_mask)
            result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(single_mask))
            result = cv2.add(result, adjusted_masked)
    
    return result

def save_processed_image(output_path, processed_image):
    """
    保存处理后的图片
    :param output_path: 输出图片路径
    :param processed_image: 处理后的图片
    """
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved as {output_path}")

# def process_images_in_folder(input_folder, output_path, alpha=1.6, beta=25, mode="full_image"):
#     """
#     遍历输入文件夹中的所有图片并处理
#     :param input_folder: 输入文件夹路径
#     :param output_path: 输出文件夹路径
#     :param alpha: 对比度因子
#     :param beta: 亮度因子
#     :param mode: 处理模式
#     """
#     create_output_path(output_path)

#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
#             image_path = os.path.join(input_folder, filename)
#             image = cv2.imread(image_path)
#             if image is None:
#                 print(f"Error: Could not load image {filename}.")
#                 continue
            
#             img_height, img_width = image.shape[:2]
#             txt_path = os.path.splitext(image_path)[0] + '.txt'
#             labels = read_labels(txt_path, output_path, img_width, img_height)
            
#             if mode in ["inside_polygon", "outside_polygon"] and not labels:
#                 print(f"Skipping {filename} due to missing labels for {mode} mode.")
#                 continue
            
#             processed_image = process_image(image, alpha, beta, mode, labels)
#             output_path = os.path.join(output_path, filename)
#             save_processed_image(output_path, processed_image)

#     print("All images have been processed.")
def process_images_in_folder(input_folder, output_path, alpha=1.6, beta=25, mode="full_image"):
    """
    遍历输入文件夹中的所有图片并处理
    :param input_folder: 输入文件夹路径
    :param output_path: 输出文件夹路径
    :param alpha: 对比度因子
    :param beta: 亮度因子
    :param mode: 处理模式
    """
    create_output_path(output_path)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {filename}.")
                continue
            
            img_height, img_width = image.shape[:2]
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            labels = read_labels(txt_path, output_path, img_width, img_height)
            
            if mode in ["inside_polygon", "outside_polygon"] and not labels:
                print(f"Skipping {filename} due to missing labels for {mode} mode.")
                continue
            
            processed_image = process_image(image, alpha, beta, mode, labels)
            # 修改这里：使用一个新的变量存储输出图片的完整路径
            output_image_path = os.path.join(output_path, filename)
            save_processed_image(output_image_path, processed_image)

    print("All images have been processed.")

def process_single_image(input_image, output_path, alpha=1.6, beta=25, mode="full_image"):
    """
    处理单张图片并保存结果
    :param input_image: 输入图片路径
    :param output_path: 输出文件夹路径
    :param alpha: 对比度因子
    :param beta: 亮度因子
    :param mode: 处理模式
    """
    create_output_path(output_path)
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: Could not load image {input_image}.")
        return
    
    img_height, img_width = image.shape[:2]
    txt_path = os.path.splitext(input_image)[0] + '.txt'
    labels = read_labels(txt_path, output_path, img_width, img_height)
    
    if mode in ["inside_polygon", "outside_polygon"] and not labels:
        print(f"Skipping {input_image} due to missing labels for {mode} mode.")
        return
    
    processed_image = process_image(image, alpha, beta, mode, labels)
    output_path = os.path.join(output_path, os.path.basename(input_image))
    save_processed_image(output_path, processed_image)

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Process images with contrast and brightness adjustment.")
    parser.add_argument('input_path', type=str, help="Path to a single image or folder of images")
    parser.add_argument('output_path', type=str, help="Path to save processed images")
    parser.add_argument('--alpha', type=float, default=3, help="Contrast factor (default: 1.6)")
    parser.add_argument('--beta', type=int, default=25, help="Brightness factor (default: 25)")
    parser.add_argument('--mode', choices=['full_image', 'inside_polygon', 'outside_polygon', 'random', 'superrandom'], default='full_image',
                        help="Processing mode (default: full_image)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    input_path = args.input_path
    output_path = args.output_path
    alpha = args.alpha
    beta = args.beta
    mode = args.mode

    if os.path.isdir(input_path):
        process_images_in_folder(input_path, output_path, alpha, beta, mode)
    elif os.path.isfile(input_path) and input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        process_single_image(input_path, output_path, alpha, beta, mode)
    else:
        print("Invalid input path. Please provide a valid image or folder.")