import cv2 as cv
import os
import math
import numpy as np
import random
from tqdm import tqdm  # 添加tqdm导入



# 1、将四点自适应扩大
def scale(img, labels):
    # 获取图像大小
    height, width, _ = img.shape
    output_labels = []

    for label in labels:
        # 依次处理每行标签
        temp = label.split(' ')
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
        det3_x = abs(int(length3 * 0.2 * math.cos(theta3)))
        det3_y = int(length3 * 0.2 * math.sin(theta3))
        p1_s[0] -= det3_x
        p4_s[0] += det3_x
        p1_s[1] -= det3_y
        p4_s[1] += det3_y

        # 横向拉长p2，p3
        length4 = pow((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1]), 0.5)
        theta4 = math.atan((p2[1] - p3[1]) / (p2[0] - p3[0]))
        det4_x = abs(int(length4 * 0.2 * math.cos(theta4)))
        det4_y = int(length4 * 0.2 * math.sin(theta4))
        p2_s[0] -= det4_x
        p3_s[0] += det4_x
        p2_s[1] -= det4_y
        p3_s[1] += det4_y

        # cv.line(img, p1_s, p2_s, color=(0, 0, 255), thickness=1)
        # cv.line(img, p2_s, p3_s, color=(0, 0, 255), thickness=1)
        # cv.line(img, p3_s, p4_s, color=(0, 0, 255), thickness=1)
        # cv.line(img, p4_s, p1_s, color=(0, 0, 255), thickness=1)

        output_labels.append(p1_s + p2_s + p3_s + p4_s)

    return output_labels

# 2、进行透视变换
# img1是哨兵图，img2为其他图，传入原来的点，扩大的点，返回变换后的图以及变换后哨兵装甲板的四点坐标
def tf(img1, label1, label1_s, img2, label2, label2_s):
    # cv.imshow('img2_ori', img2)
    output_labels = []
    for i in range(len(label2_s)):
        temp1 = np.float32([[label1_s[0][0], label1_s[0][1]], [label1_s[0][2], label1_s[0][3]],
                            [label1_s[0][4], label1_s[0][5]], [label1_s[0][6], label1_s[0][7]]])

        temp2 = np.float32([[label2_s[i][0], label2_s[i][1]], [label2_s[i][2], label2_s[i][3]],
                            [label2_s[i][4], label2_s[i][5]], [label2_s[i][6], label2_s[i][7]]])

        matrix = cv.getPerspectiveTransform(temp1, temp2)

        height2, width2, _ = img2.shape
        height1, width1, _ = img1.shape
        result = cv.warpPerspective(img1, matrix, (width2, height2))

        # 制作掩膜
        mask = np.zeros((height2, width2), dtype=np.uint8)
        # 绘制白色矩形
        points = np.array(np.expand_dims(temp2, axis=0), dtype=np.int32)
        cv.polylines(mask, points, True, (255, 255, 255, 0))
        # 填充矩形
        cv.fillPoly(mask, points, (255, 255, 255))
        # 覆盖
        cv.copyTo(result, mask, img2)

        # 将标签通过投射变换矩阵进行变换
        folder_label = label1[0].split(' ')
        folder_label = np.array(folder_label[1:], np.float32)
        folder_label *= ([width1, height1] * 4)
        folder_label = folder_label.reshape(4, 2)
        new_col = [1, 1, 1, 1]
        folder_label = np.column_stack((folder_label, new_col))
        folder_label = np.transpose(folder_label)
        tf_label = np.dot(matrix, folder_label)
        tf_label = np.transpose(tf_label)
        tf_label[:, :2] /= tf_label[:, 2:3]
        tf_label = np.array(tf_label[:, :2], np.int32)

        # 新造一个字符串用来写入
        newboy = ""
        for boy in tf_label:
            newboy = newboy + str(round(boy[0] / width2, 6)) + " " + str(round(boy[1] / height2, 6)) + " "
        # 去掉最后一个空格
        newboy = newboy[:-1]

        output_labels.append(newboy)

        # cv.line(img2, tf_label[0], tf_label[1], color=(0, 255, 0), thickness=1)
        # cv.line(img2, tf_label[1], tf_label[2], color=(0, 255, 0), thickness=1)
        # cv.line(img2, tf_label[2], tf_label[3], color=(0, 255, 0), thickness=1)
        # cv.line(img2, tf_label[3], tf_label[0], color=(0, 255, 0), thickness=1)

    # cv.imshow('img1', img1)
    # cv.imshow('img2', img2)
    # cv.waitKey(0)

    return img2, output_labels

# 3、对已有哨兵进行替换
# 接收三个地址，保存到最后一个地址
def steal(dir1, dir2, dir3):
    # 拿出哨兵数据
    sentry_img_files = []
    sentry_label_files = []
    files = os.listdir(dir1)
    for file in files:
        name, suf = file.split(".")
        if suf == 'txt':
            sentry_label_files.append(file)
        else:
            sentry_img_files.append(file)


        # 拿出其他数据
    other_img_files = []
    other_label_files = []
    new_files = os.listdir(dir2)
    for new_file in new_files:
        if os.path.isfile(os.path.join(dir2, new_file)):
            name, suf = new_file.split(".")
            if suf == 'txt':
                other_label_files.append(new_file)
            else:
                other_img_files.append(new_file)

    # 遍历哨兵数据
    for i in tqdm(range(len(sentry_img_files)), desc="处理进度"):
        # 获取哨兵图片和标签
        sentry_img = cv.imread(os.path.join(dir1, sentry_img_files[i]))
        sentry_label = []
        temp_name, _ = sentry_img_files[i].split('.')
        if os.path.exists(os.path.join(dir1, temp_name + '.txt')) == False:
            continue
        with open(os.path.join(dir1, temp_name + '.txt')) as temp_file:
            # 只拿一个哨兵装甲板
            for line in temp_file:
                sentry_label.append(line)
                break
        # 获得自适应扩大的标签
        sentry_label_out = scale(sentry_img, sentry_label)

        # 获得该哨兵装甲板标签（如果为空标签则跳过）
        if len(sentry_label) == 0:
            continue
        temp_boys = sentry_label[0].split(' ')
        boy = temp_boys[0]

        # 获取其他图片和标签
        while True:
            # 获取下标
            temp_key = random.randint(0, len(other_img_files) - 1)
            # 排除掉大装甲板的情况
            temp_name1, _ = other_img_files[temp_key].split('.')
            cnt = 0
            with open(os.path.join(dir2, temp_name1 + '.txt')) as temp_file:
                for line in temp_file:
                    temp = line.split(' ')
                    if temp[0] >= '23':
                        cnt += 1
            # 如果没有大装甲板，则可以使用
            if cnt == 0:
                break

        other_img = cv.imread(os.path.join(dir2, other_img_files[temp_key]))
        other_label = []

        with open(os.path.join(dir2, temp_name1 + '.txt')) as temp_file:
            # 将数据都拿出
            for line in temp_file:
                other_label.append(line)
        # 获得自适应扩大的标签
        other_label_out = scale(other_img, other_label)

        # 进行替换
        img_out, output_labels = tf(sentry_img, sentry_label, sentry_label_out, other_img, other_label, other_label_out)

        # 保存数据
        cv.imwrite(dir3 + '/' + str(i) + '.jpg', img_out)
        for j in range(len(output_labels)):
            output_labels[j] = boy + ' ' + output_labels[j]
        # 将更改后的标签写入新label文件
        with open(dir3 + '/' + str(i) + '.txt', 'w') as tem:
            for line in output_labels:
                tem.write(line)



# img1_dir = "/home/horsefly/armor_det/armorDet/mine/resource/HERO-23-OTH-83.jpg"
# label1_dir = "/home/horsefly/armor_det/armorDet/mine/resource/HERO-23-OTH-83.txt"
# img1 = cv.imread(img1_dir)
# labels1 = []
# with open(label1_dir, 'r') as file:
#     for line in file:
#         labels1.append(line)
# output_labels1 = scale(img1, labels1)
#
# img2_dir = "/home/horsefly/armor_det/armorDet/mine/resource/136.png"
# label2_dir = "/home/horsefly/armor_det/armorDet/mine/resource/136.txt"
# img2 = cv.imread(img2_dir)
# labels2 = []
# with open(label2_dir, 'r') as file:
#     for line in file:
#         labels2.append(line)
# output_labels2 = scale(img2, labels2)
#
# tf(img1, labels1, output_labels1, img2, labels2, output_labels2)

# dir1 = "/home/horsefly/下载/hero哨兵/train"
# dir1 = "/home/horsefly/下载/北邮新哨兵/images/train"
dir1 = "/home/test/Desktop/Data_Enhance/images/new_sentry"
dir2 = "/home/test/Desktop/Data_Enhance/images/temp"
dir3 = "/home/test/Desktop/Data_Enhance/output"

steal(dir1, dir2, dir3)

