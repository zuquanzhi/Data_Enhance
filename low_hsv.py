import cv2 as cv
import os
import math
import numpy as np
import random
import argparse


# 将四点自适应扩大
def scale(img, labels):
    # 获取图像大小
    height, width, _ = img.shape
    output_labels = []

    for label in labels:
        flag = True

        # 依次处理每行标签，如果不是前哨站则跳过
        temp = label.split(' ')
        if temp[0] != "18" and temp[0] != "19" and temp[0] != "20":
            flag = False

        # 如果是前哨站但是标签有问题也跳过
        for i in range(len(temp)):
            if temp[i] == "" and len(temp) == 9:
                flag = False

        if flag == False:  ##前哨站开关
            continue

        for i in range(len(temp)):
            temp[i] = float(temp[i])

        # 将坐标化为绝对坐标
        xyxyxyxy = list(map(lambda x, y: x * y, temp[1:], [width, height, width, height, width, height, width, height]))

        p1 = list(map(int, xyxyxyxy[0: 2]))
        p2 = list(map(int, xyxyxyxy[2: 4]))
        p3 = list(map(int, xyxyxyxy[4: 6]))
        p4 = list(map(int, xyxyxyxy[6: 8]))

        # cv.line(img, p1, p2, color=(0, 255, 0), thickness=1)
        # cv.line(img, p2, p3, color=(0, 255, 0), thickness=1)
        # cv.line(img, p3, p4, color=(0, 255, 0), thickness=1)
        # cv.line(img, p4, p1, color=(0, 255, 0), thickness=1)

        p1_s = p1
        p2_s = p2
        p3_s = p3
        p4_s = p4

        # 纵向拉长p1，p2
        length1 = pow((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]), 0.5)
        if p2[0] - p1[0] != 0:
            theta1 = math.atan((p2[1] - p1[1]) / (p2[0] - p1[0]))
        else:
            theta1 = 3.1415926 / 2
        det1_y = abs(int(length1 * 0.5 * math.sin(theta1)))
        det1_x = int(theta1 / abs(theta1) * length1 * 0.5 * math.cos(theta1))
        p1_s[1] -= det1_y
        p2_s[1] += det1_y
        p1_s[0] -= det1_x
        p2_s[0] += det1_x

        # 纵向拉长p4，p3
        length2 = pow((p4[0] - p3[0]) * (p4[0] - p3[0]) + (p4[1] - p3[1]) * (p4[1] - p3[1]), 0.5)
        if p3[0] - p4[0] != 0:
            theta2 = math.atan((p3[1] - p4[1]) / (p3[0] - p4[0]))
        else:
            theta2 = 3.1415926 / 2
        det2_y = abs(int(length2 * 0.5 * math.sin(theta2)))
        det2_x = int(theta2 / abs(theta2) * length2 * 0.5 * math.cos(theta2))
        p4_s[1] -= det2_y
        p3_s[1] += det2_y
        p4_s[0] -= det2_x
        p3_s[0] += det2_x

        # cv.line(img, p1_s, p2_s, color=(255, 255, 0), thickness=1)
        # cv.line(img, p2_s, p3_s, color=(255, 255, 0), thickness=1)
        # cv.line(img, p3_s, p4_s, color=(255, 255, 0), thickness=1)
        # cv.line(img, p4_s, p1_s, color=(255, 255, 0), thickness=1)

        # 横向拉长p1，p4
        length3 = pow((p1[0] - p4[0]) * (p1[0] - p4[0]) + (p1[1] - p4[1]) * (p1[1] - p4[1]), 0.5)
        theta3 = math.atan((p1[1] - p4[1]) / (p1[0] - p4[0]))
        det3_x = abs(int(length3 * 0.1 * math.cos(theta3)))
        det3_y = int(length3 * 0.2 * math.sin(theta3))
        p1_s[0] += det3_x
        p4_s[0] -= det3_x
        p1_s[1] -= det3_y
        p4_s[1] += det3_y

        # 横向拉长p2，p3
        length4 = pow((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1]), 0.5)
        theta4 = math.atan((p2[1] - p3[1]) / (p2[0] - p3[0]))
        det4_x = abs(int(length4 * 0.1 * math.cos(theta4)))
        det4_y = int(length4 * 0.2 * math.sin(theta4))
        p2_s[0] += det4_x
        p3_s[0] -= det4_x
        p2_s[1] -= det4_y
        p3_s[1] += det4_y

        # cv.line(img, p1_s, p2_s, color=(0, 0, 255), thickness=1)
        # cv.line(img, p2_s, p3_s, color=(0, 0, 255), thickness=1)
        # cv.line(img, p3_s, p4_s, color=(0, 0, 255), thickness=1)
        # cv.line(img, p4_s, p1_s, color=(0, 0, 255), thickness=1)

        output_labels.append(p1_s + p2_s + p3_s + p4_s)
        # cv.imwrite("/home/horsefly/下载/temp/d.jpg", img)

    return output_labels

# 将图片装甲板中间部分进行随机亮度降低
def low_hsv(input_path,output_path):
    # 拿出数据
    img_files = []
    label_files = []
    files = os.listdir(input_path)
    for file in files:
        name, suf = file.split(".")
        if suf == 'txt':
            label_files.append(file)
        else:
            img_files.append(file)

    # 遍历数据
    for i in range(len(img_files)):
        # 获取图片和标签
        img = cv.imread(os.path.join(input_path, img_files[i]))
        label = []
        temp_name, _ = img_files[i].split('.')
        if os.path.exists(os.path.join(input_path, temp_name + '.txt')) == False:
            continue
        with open(os.path.join(input_path, temp_name + '.txt'), "r") as temp_file:
            for line in temp_file:
                label.append(line)

        output_label_path = os.path.join(output_path, f"{temp_name}.txt")
        with open(output_label_path,'w') as f:
            for line in label:
                f.write(line)

        # 获得自适应扩大的标签
        label_out = scale(img, label)

        # 对每个自适应扩大的标签区域进行随机亮度降低
        for i in range(len(label_out)):
            temp = np.float32([[label_out[i][0], label_out[i][1]], [label_out[i][2], label_out[i][3]],
                                [label_out[i][4], label_out[i][5]], [label_out[i][6], label_out[i][7]]])

            height, width, _ = img.shape
            # 制作掩膜
            mask = np.zeros((height, width), dtype=np.uint8)
            # 绘制白色矩形
            points = np.array(np.expand_dims(temp, axis=0), dtype=np.int32)
            cv.polylines(mask, points, True, (255, 255, 255, 0))
            # 填充矩形
            cv.fillPoly(mask, points, (255, 255, 255))
            # 覆盖
            roi = np.zeros((height, width, 3), dtype=np.uint8)
            cv.copyTo(img, mask, roi)
            roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            # 对roi进行随机亮度下降
            v = np.random.uniform(0.08, 0.3)
            value = roi[:, :, 2]
            new_value = value * v
            new_value = new_value.astype(np.uint8)
            new_value[new_value > 255] = 255
            new_value[new_value < 0] = 0
            roi[:, :, 2] = new_value
            roi = cv.cvtColor(roi, cv.COLOR_HSV2BGR)
            # 再利用mask将roi覆盖回去
            cv.copyTo(roi, mask, img)

        # 将图像和标签进行保存
        output_image_path = os.path.join(output_path, f"{temp_name}.jpg")
        cv.imwrite(output_image_path, img)


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Low the brightness of the middle part of the armor plate")
    parser.add_argument('input_path', type=str, help="Path to a single image or folder of images")
    parser.add_argument('output_path', type=str, help="Path to save processed images")
    return parser.parse_args()


def process_single_image(input_image, output_path):
    """处理单张图片
    :param input_image: 输入图片路径
    :param output_path: 输出文件夹路径
    """
    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 读取图片
    img = cv.imread(input_image)
    if img is None:
        print(f"Error: Could not read image {input_image}")
        return
        
    # 获取标签文件路径
    temp_name = os.path.splitext(os.path.basename(input_image))[0]
    txt_path = os.path.splitext(input_image)[0] + '.txt'
    
    # 读取标签
    label = []
    if not os.path.exists(txt_path):
        print(f"Warning: Label file {txt_path} does not exist.")
        return
    
    with open(txt_path, "r") as temp_file:
        label = temp_file.readlines()
    
    # 保存标签到输出目录
    output_label_path = os.path.join(output_path, f"{temp_name}.txt")
    with open(output_label_path, 'w') as f:
        f.writelines(label)
    
    # 处理图像
    label_out = scale(img, label)
    
    # 对每个区域进行处理
    for coords in label_out:
        temp = np.float32([
            [coords[0], coords[1]], [coords[2], coords[3]],
            [coords[4], coords[5]], [coords[6], coords[7]]
        ])
        
        height, width, _ = img.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array(np.expand_dims(temp, axis=0), dtype=np.int32)
        cv.fillPoly(mask, points, (255, 255, 255))
        
        roi = np.zeros((height, width, 3), dtype=np.uint8)
        cv.copyTo(img, mask, roi)
        roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        
        v = np.random.uniform(0.08, 0.3)
        roi[:, :, 2] = (roi[:, :, 2] * v).clip(0, 255).astype(np.uint8)
        roi = cv.cvtColor(roi, cv.COLOR_HSV2BGR)
        cv.copyTo(roi, mask, img)
    
    # 保存处理后的图片
    output_image_path = os.path.join(output_path, f"{temp_name}.jpg")
    cv.imwrite(output_image_path, img)
    print(f"Processed and saved: {output_image_path}")

# 修改main部分
if __name__ == "__main__":
    args = parse_arguments()
    input_path = args.input_path
    output_path = args.output_path

    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.isdir(input_path):
        # 处理文件夹
        low_hsv(input_path, output_path)
    elif os.path.isfile(input_path) and input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # 处理单张图片
        process_single_image(input_path, output_path)
    else:
        print("Error: Input path must be an image file or directory")


# if __name__ == "__main__":
#     args = parse_arguments()
#     input_path = args.input_path
#     output_path = args.output_path

#     if os.path.isdir(input_path):
#         low_hsv(input_path, output_path)
#     else:
#         print("The input path is not a folder")