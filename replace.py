import cv2 as cv
import os
import math
import numpy as np
import random
import argparse
from tqdm import tqdm

def adjust_points(point1, point2, length_ratio=0.5):
    """调整两点的位置
    Args:
        point1: 第一个点的坐标 [x, y]
        point2: 第二个点的坐标 [x, y]
        length_ratio: 延长比例，默认0.5
    Returns:
        point1_adjusted, point2_adjusted: 调整后的两点坐标
    """
    point1_adjusted = point1.copy()
    point2_adjusted = point2.copy()
    
    length = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    if point2[0] - point1[0] != 0:
        angle = math.atan((point2[1] - point1[1]) / (point2[0] - point1[0]))
    else:
        angle = math.pi / 2
        
    det_y = abs(int(length * length_ratio * math.sin(angle)))
    det_x = int(angle / abs(angle) * length * length_ratio * math.cos(angle))
    
    point1_adjusted[1] -= det_y
    point2_adjusted[1] += det_y
    point1_adjusted[0] -= det_x
    point2_adjusted[0] += det_x
    
    return point1_adjusted, point2_adjusted


def scale(image, labels, vertical_ratio=0.5, horizontal_ratio=0.2):
    """将四点自适应扩大
    Args:
        image: 输入图像
        labels: 标注数据
        vertical_ratio: 纵向扩展比例，默认0.5
        horizontal_ratio: 横向扩展比例，默认0.2
    Returns:
        adjusted_labels: 调整后的标注数据
    """
    height, width, _ = image.shape
    adjusted_labels = []
    
    for label in labels:
        temp = label.split(' ')
        temp = [float(x) for x in temp]
        
        # 将坐标化为绝对坐标
        coords = list(map(lambda x, y: x * y, temp[1:], [width, height] * 4))
        points = [list(map(int, coords[i:i+2])) for i in range(0, 8, 2)]
        
        # 复制点以防修改原始数据
        points_adjusted = [p.copy() for p in points]
        
        # 纵向拉长point1-point2和point4-point3
        points_adjusted[0], points_adjusted[1] = adjust_points(points_adjusted[0], points_adjusted[1], vertical_ratio)
        points_adjusted[3], points_adjusted[2] = adjust_points(points_adjusted[3], points_adjusted[2], vertical_ratio)
        
        # 横向拉长point1-point4和point2-point3
        points_adjusted[0], points_adjusted[3] = adjust_points(points_adjusted[0], points_adjusted[3], horizontal_ratio)
        points_adjusted[1], points_adjusted[2] = adjust_points(points_adjusted[1], points_adjusted[2], horizontal_ratio)
        
        # 合并调整后的坐标
        adjusted_coords = []
        for p in points_adjusted:
            adjusted_coords.extend(p)
            
        adjusted_labels.append(adjusted_coords)
        
    return adjusted_labels

# 进行透视变换
def perspective_transform(image1, label1, adjusted_labels1, image2, label2, adjusted_labels2):
    output_labels = []
    for i in range(len(adjusted_labels2)):
        points1 = np.float32([[adjusted_labels1[0][0], adjusted_labels1[0][1]], [adjusted_labels1[0][2], adjusted_labels1[0][3]],
                            [adjusted_labels1[0][4], adjusted_labels1[0][5]], [adjusted_labels1[0][6], adjusted_labels1[0][7]]])
        points2 = np.float32([[adjusted_labels2[i][0], adjusted_labels2[i][1]], [adjusted_labels2[i][2], adjusted_labels2[i][3]],
                            [adjusted_labels2[i][4], adjusted_labels2[i][5]], [adjusted_labels2[i][6], adjusted_labels2[i][7]]])
        matrix = cv.getPerspectiveTransform(points1, points2)
        height2, width2, _ = image2.shape
        result = cv.warpPerspective(image1, matrix, (width2, height2))
        # 制作掩膜
        mask = np.zeros((height2, width2), dtype=np.uint8)
        # 绘制白色矩形
        points = np.array(np.expand_dims(points2, axis=0), dtype=np.int32)
        cv.polylines(mask, points, True, (255, 255, 255, 0))
        # 填充矩形
        cv.fillPoly(mask, points, (255, 255, 255))
        # 覆盖
        cv.copyTo(result, mask, image2)
        # 将标签通过投射变换矩阵进行变换
        label_coords = label1[0].split(' ')
        label_coords = np.array(label_coords[1:], np.float32)
        label_coords *= ([image1.shape[1], image1.shape[0]] * 4)
        label_coords = label_coords.reshape(4, 2)
        new_col = [1, 1, 1, 1]
        label_coords = np.column_stack((label_coords, new_col))
        label_coords = np.transpose(label_coords)
        transformed_label = np.dot(matrix, label_coords)
        transformed_label = np.transpose(transformed_label)
        transformed_label[:, :2] /= transformed_label[:, 2:3]
        transformed_label = np.array(transformed_label[:, :2], np.int32)
        new_label = ""
        for point in transformed_label:
            new_label += str(round(point[0] / width2, 6)) + " " + str(round(point[1] / height2, 6)) + " "
        new_label = new_label[:-1]
        output_labels.append(new_label)
    return image2, output_labels

# 3. 对已有目标对象进行替换

def replace_object_data(input_dir, reference_dir, output_dir, start_index=100):
    """对已有目标对象进行替换"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 拿出目标对象数据
    object_image_files = []
    object_label_files = []
    files = os.listdir(input_dir)
    for file in files:
        name, suf = file.split(".")
        if suf == 'txt':
            object_label_files.append(file)
        else:
            object_image_files.append(file)
    
    # 拿出其他数据
    other_image_files = []
    other_label_files = []
    new_files = os.listdir(reference_dir)
    for new_file in new_files:
        if os.path.isfile(os.path.join(reference_dir, new_file)):
            name, suf = new_file.split(".")
            if suf == 'txt':
                other_label_files.append(new_file)
            else:
                other_image_files.append(new_file)
    
    # 创建进度条
    pbar = tqdm(total=len(object_image_files), 
                desc="处理进度",
                ncols=100)
    
    # 遍历目标对象数据
    for i in range(len(object_image_files)):
        # 获取目标对象图片和标签
        object_img = cv.imread(os.path.join(input_dir, object_image_files[i]))
        object_label = []
        temp_name, _ = object_image_files[i].split('.')
        
        if not os.path.exists(os.path.join(input_dir, temp_name + '.txt')):
            pbar.update(1)
            continue
            
        with open(os.path.join(input_dir, temp_name + '.txt')) as temp_file:
            # 只拿一个目标对象
            for line in temp_file:
                object_label.append(line)
                break
                
        object_label_adjusted = scale(object_img, object_label)
        
        if len(object_label) == 0:
            pbar.update(1)
            continue
            
        temp_boys = object_label[0].split(' ')
        boy = temp_boys[0]
        
        # 查找合适的背景图片
        while True:
            temp_key = random.randint(0, len(other_image_files) - 1)
            temp_name1, _ = other_image_files[temp_key].split('.')
            cnt = 0
            with open(os.path.join(reference_dir, temp_name1 + '.txt')) as temp_file:
                for line in temp_file:
                    temp = line.split(' ')
                    if temp[0] >= '23':
                        cnt += 1
            if cnt == 0:
                break
        
        # 读取背景图片和标注
        other_img = cv.imread(os.path.join(reference_dir, other_image_files[temp_key]))
        other_label = []
        with open(os.path.join(reference_dir, temp_name1 + '.txt')) as temp_file:
            for line in temp_file:
                other_label.append(line)
                
        other_label_adjusted = scale(other_img, other_label)
        
        # 透视变换
        img_out, output_labels = perspective_transform(
            object_img, object_label, object_label_adjusted,
            other_img, other_label, other_label_adjusted
        )
        
        # 保存结果
        output_index = start_index + i
        cv.imwrite(os.path.join(output_dir, f'{output_index}.jpg'), img_out)
        
        # 更新标签
        for j in range(len(output_labels)):
            output_labels[j] = boy + ' ' + output_labels[j]
            
        # 保存标注文件
        with open(os.path.join(output_dir, f'{output_index}.txt'), 'w') as temp_file:
            for line in output_labels:
                temp_file.write(line + '\n')
                
        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({"当前文件": object_image_files[i]})
    
    # 关闭进度条
    pbar.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据增强工具')
    parser.add_argument('--input', type=str, default='./target',help='输入目录路径,包含待处理的目标图像和标注')
    parser.add_argument('--reference', type=str, default='./images',help='参考图像目录路径,用于替换背景')
    parser.add_argument('--output', type=str, default='./output',help='输出目录路径')
    parser.add_argument('--start-index', type=int, default=250,help='输出文件的起始编号')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 执行数据增强
    replace_object_data(
        args.input,
        args.reference,
        args.output,
        args.start_index
    )

if __name__ == "__main__":
    main()