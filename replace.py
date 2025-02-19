import cv2 as cv
import os
import math
import numpy as np
import random


# 1. 将四点自适应扩大
def scale(img, labels, vertical_scale_factor=0.5, horizontal_scale_factor=0.2):
    """
    自适应扩大标签的四点坐标
    :param img: 输入图像
    :param labels: 图像的标签
    :param vertical_scale_factor: 纵向拉伸的比例
    :param horizontal_scale_factor: 横向拉伸的比例
    :return: 扩大的标签
    """
    height, width, _ = img.shape
    output_labels = []

    for label in labels:
        temp = label.split(' ')
        temp = [float(x) for x in temp]

        # 将坐标化为绝对坐标
        xyxyxyxy = list(map(lambda x, y: x * y, temp[1:], [width, height, width, height, width, height, width, height]))

        p1, p2, p3, p4 = [list(map(int, xyxyxyxy[i:i+2])) for i in range(0, 8, 2)]
        
        # 缩放处理
        p1_s, p2_s, p3_s, p4_s = p1.copy(), p2.copy(), p3.copy(), p4.copy()

        # 纵向和横向拉伸处理
        for p1, p2, scale_factor in [(p1, p2, vertical_scale_factor), (p4, p3, vertical_scale_factor),(p1, p4, horizontal_scale_factor), (p2, p3, horizontal_scale_factor)]:
            length = np.linalg.norm(np.subtract(p1, p2))
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) if p2[0] != p1[0] else math.pi / 2
            det_x = int(scale_factor * length * math.cos(angle))
            det_y = int(scale_factor * length * math.sin(angle))

            p1_s[0] -= det_x
            p1_s[1] -= det_y
            p2_s[0] += det_x
            p2_s[1] += det_y

        output_labels.append(p1_s + p2_s + p3_s + p4_s)

    return output_labels


# 2. 进行透视变换
def tf(img1, label1, label1_s, img2, label2, label2_s):
    """
    执行透视变换并替换图像
    :param img1: 源图像
    :param label1: 源图像的标签
    :param label1_s: 源图像的扩展标签
    :param img2: 目标图像
    :param label2: 目标图像的标签
    :param label2_s: 目标图像的扩展标签
    :return: 变换后的图像和标签
    """
    output_labels = []

    for i in range(len(label2_s)):
        temp1 = np.float32(np.array(label1_s[0]).reshape(4, 2))
        temp2 = np.float32(np.array(label2_s[i]).reshape(4, 2))

        matrix = cv.getPerspectiveTransform(temp1, temp2)

        height2, width2, _ = img2.shape
        result = cv.warpPerspective(img1, matrix, (width2, height2))

        # 制作掩膜
        mask = np.zeros((height2, width2), dtype=np.uint8)
        points = np.array(np.expand_dims(temp2, axis=0), dtype=np.int32)
        cv.polylines(mask, points, True, (255, 255, 255, 0))
        cv.fillPoly(mask, points, (255, 255, 255))
        cv.copyTo(result, mask, img2)

        # 将标签通过投射变换矩阵进行变换
        folder_label = label1[0].split(' ')
        folder_label = np.array(folder_label[1:], np.float32)
        folder_label *= ([width2, height2] * 4)
        folder_label = folder_label.reshape(4, 2)
        new_col = [1, 1, 1, 1]
        folder_label = np.column_stack((folder_label, new_col))
        folder_label = np.transpose(folder_label)
        tf_label = np.dot(matrix, folder_label)
        tf_label = np.transpose(tf_label)
        tf_label[:, :2] /= tf_label[:, 2:3]
        tf_label = np.array(tf_label[:, :2], np.int32)

        newboy = " ".join([f"{round(boy[0] / width2, 6)} {round(boy[1] / height2, 6)}" for boy in tf_label])
        output_labels.append(newboy)

    return img2, output_labels


# 3. 替换哨兵数据
def steal(src_dir, dst_dir, output_dir, start_index=100, min_category_id=23):
    print(f"开始处理源图像：{src_dir} 和目标图像：{dst_dir}，输出目录：{output_dir}，起始索引：{start_index}")

    sentry_img_files = [file for file in os.listdir(src_dir) if file.lower().endswith(('.jpg', '.png'))]
    sentry_label_files = [file for file in os.listdir(src_dir) if file.lower().endswith('.txt')]
    other_img_files = [file for file in os.listdir(dst_dir) if file.lower().endswith(('.jpg', '.png'))]
    other_label_files = [file for file in os.listdir(dst_dir) if file.lower().endswith('.txt')]

    print(f"共找到 {len(sentry_img_files)} 个源图像，{len(other_img_files)} 个目标图像")

    for i, img_file in enumerate(sentry_img_files):
        print(f"处理第 {i + 1} 个图像...")

        sentry_img = cv.imread(os.path.join(src_dir, img_file))
        sentry_label = []
        temp_name = os.path.splitext(img_file)[0] + '.txt'

        if not os.path.exists(os.path.join(src_dir, temp_name)):
            print(f"跳过 {img_file}：缺少标签文件")
            continue

        with open(os.path.join(src_dir, temp_name)) as f:
            sentry_label = [line.strip() for line in f.readlines()]

        sentry_label_out = scale(sentry_img, sentry_label)

        temp_boys = sentry_label[0].split(' ')
        boy = temp_boys[0]

        # 随机选择目标图像
        selected_dst_img_file = None
        for _ in range(10):
            selected_dst_img_file = random.choice(other_img_files)
            dst_label_file = os.path.splitext(selected_dst_img_file)[0] + '.txt'

            if os.path.exists(os.path.join(dst_dir, dst_label_file)):
                with open(os.path.join(dst_dir, dst_label_file)) as f:
                    lines = [line.strip().split(' ')[0] for line in f.readlines()]
                    if all(int(cls) < min_category_id for cls in lines):
                        break

        dst_img = cv.imread(os.path.join(dst_dir, selected_dst_img_file))
        dst_label = []
        with open(os.path.join(dst_dir, dst_label_file)) as f:
            dst_label = [line.strip() for line in f.readlines()]

        dst_label_out = scale(dst_img, dst_label)

        img_out, output_labels = tf(sentry_img, sentry_label, sentry_label_out, dst_img, dst_label, dst_label_out)

        # 保存结果
        output_index = i + start_index
        cv.imwrite(os.path.join(output_dir, f'{output_index}.jpg'), img_out)

        with open(os.path.join(output_dir, f'{output_index}.txt'), 'w') as f:
            for label in output_labels:
                f.write(f"{boy} {label}\n")

        print(f"图像 {img_file} 处理完成，保存为 {output_index}.jpg")

    print("所有图像处理完成！")


# 输入配置
src_dir = "/home/test/Desktop/Tools/test/dst"  # 替换为你的源图像目录
dst_dir = "/home/test/Desktop/Tools/test/src"  # 替换为你的目标图像目录
output_dir = "/home/test/Desktop/Tools/test/output"  # 替换为你的输出目录
start_index = 100  # 起始索引
min_category_id = 23  # 跳过目标图像中类别ID大于或等于该值的图像

# 执行处理
steal(src_dir, dst_dir, output_dir, start_index, min_category_id)
