import cv2 as cv
import os
import numpy as np

def visualize_annotations(image_path, label_path):
    """
    读取图像和同名的标注文件，显示标注点并连接。
    :param image_path: 图像文件路径
    :param label_path: 标注文件路径
    """
    # 读取图像
    img = cv.imread(image_path)
    
    if img is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 读取标注文件
    if not os.path.exists(label_path):
        print(f"标注文件不存在: {label_path}")
        return
    
    with open(label_path, 'r') as f:
        for line in f:
            # 解析标注文件中的坐标
            parts = line.strip().split()
            class_id = parts[0]
            # 提取坐标信息，转换为浮动坐标，然后映射到图像大小
            coords = [float(x) for x in parts[1:]]
            
            # 假设每四个点构成一个多边形，先将坐标转为整数
            points = [(int(coords[i] * img.shape[1]), int(coords[i+1] * img.shape[0])) for i in range(0, len(coords), 2)]
            
            # 绘制每四个点为一个图形
            for i in range(0, len(points), 4):
                # 取四个点
                quadrilateral = points[i:i+4]
                
                # 绘制多边形
                for j in range(len(quadrilateral) - 1):
                    cv.line(img, quadrilateral[j], quadrilateral[j+1], color=(0, 255, 0), thickness=1)
                
                # 如果是闭合的标注，最后一个点和第一个点连接
                if len(quadrilateral) == 4:
                    cv.line(img, quadrilateral[-1], quadrilateral[0], color=(0, 255, 0), thickness=1)
            
            # 绘制每个点
            for point in points:
                cv.circle(img, point, 3, (255, 0, 0), -1)
            
            # 计算多边形的中心点，以便显示标签
            if len(points) >= 4:
                center = np.mean(points[:4], axis=0).astype(int)
                # 显示class_id
                cv.putText(img, class_id, tuple(center), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

    # 显示带标注的图像
    cv.imshow("Image with annotations", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    # 输入图像和标注文件的路径
    image_path = './dataset/24_fq_sy_outpost17.jpg'
    label_path = './dataset/24_fq_sy_outpost17.txt'
    
    # 调用可视化函数
    visualize_annotations(image_path, label_path)

