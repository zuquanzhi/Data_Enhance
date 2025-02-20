import os
from PIL import Image, ImageFilter

# 应用运动模糊效果
def apply_motion_blur(img, blur_radius):
    # 使用GaussianBlur模拟模糊效果
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

# 处理单个图像文件
def process_single_image(input_image_path, output_directory, blur_radius):
    # 打开图片
    img = Image.open(input_image_path)
    
    # 应用运动模糊
    blurred_img = apply_motion_blur(img, blur_radius)
    
    # 生成输出路径
    output_path = os.path.join(output_directory, os.path.basename(input_image_path))

    input_txt_path = input_image_path.replace(os.path.splitext(input_image_path)[1], '.txt')
    generate_yolo_label(input_txt_path, output_path.replace(os.path.splitext(output_path)[1], '.txt'))
    
    # 保存模糊后的图片
    blurred_img.save(output_path)
    print(f"处理完成: {input_image_path} -> {output_path}")

# 处理文件夹内所有图像文件
def process_images_in_directory(input_directory, output_directory, blur_radius):
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_directory):
        input_image_path = os.path.join(input_directory, filename)

        # 仅处理图像文件（可以根据需要添加更多格式）
        if os.path.isfile(input_image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_single_image(input_image_path, output_directory, blur_radius)

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



# 主函数，支持单个文件和文件夹输入
def main(input_path, output_directory, blur_radius):
    # 检查输入路径是文件还是文件夹
    if os.path.isdir(input_path):
        process_images_in_directory(input_path, output_directory, blur_radius)
    elif os.path.isfile(input_path):
        process_single_image(input_path, output_directory, blur_radius)
    else:
        print("无效的输入路径!")

# 用户设置
input_path = '/home/test/Desktop/Data_Enhance/images/src'  # 输入路径，可以是文件夹或单个图像文件
output_directory = 'output'  # 输出路径（文件夹）
blur_radius = 5  # 设置运动模糊的程度（模糊半径）

# 调用主函数
main(input_path, output_directory, blur_radius)
