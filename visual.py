import cv2 as cv
import os
import numpy as np

def visualize_annotations(image_folder):
    """
    遍历文件夹内的所有图像和同名的标注文件，显示标注点并连接。
    支持方向键切换图片，按 q 退出。
    :param image_folder: 包含图像和标注文件的文件夹路径
    """
    # 获取文件夹中所有的图片文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()  # 按文件名排序，确保顺序一致
    
    if not image_files:
        print("文件夹中没有找到任何图片文件。")
        return
    
    current_index = 0  # 当前显示的图片索引
    total_images = len(image_files)
    
    while True:
        # 获取当前图片和标注文件的路径
        image_file = image_files[current_index]
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(image_folder, os.path.splitext(image_file)[0] + ".txt")
        
        # 读取图像
        img = cv.imread(image_path)
        if img is None:
            print(f"无法加载图像: {image_path}")
            current_index = (current_index + 1) % total_images  # 跳过无效图片
            continue
        
        # 清空图像上的标注
        img_copy = img.copy()
        
        # 检查标注文件是否存在
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # 解析标注文件中的坐标
                    parts = line.strip().split()
                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:]]
                    
                    # 假设每四个点构成一个多边形，先将坐标转为整数
                    points = [(int(coords[i] * img.shape[1]), int(coords[i+1] * img.shape[0])) for i in range(0, len(coords), 2)]
                    
                    # 绘制每四个点为一个图形
                    for i in range(0, len(points), 4):
                        quadrilateral = points[i:i+4]
                        
                        # 绘制多边形
                        for j in range(len(quadrilateral) - 1):
                            cv.line(img_copy, quadrilateral[j], quadrilateral[j+1], color=(0, 255, 0), thickness=1)
                        
                        # 如果是闭合的标注，最后一个点和第一个点连接
                        if len(quadrilateral) == 4:
                            cv.line(img_copy, quadrilateral[-1], quadrilateral[0], color=(0, 255, 0), thickness=1)
                    
                    # 绘制每个点
                    for point in points:
                        cv.circle(img_copy, point, 3, (255, 0, 0), -1)
                    
                    # 计算多边形的中心点，以便显示标签
                    if len(points) >= 4:
                        center = np.mean(points[:4], axis=0).astype(int)
                        offset_x, offset_y = 10, -10  # 偏移量
                        label_position = (center[0] + offset_x, center[1] + offset_y)
                        cv.putText(img_copy, class_id, label_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        
        # 显示带标注的图像
        cv.imshow("Image with annotations", img_copy)
        
        # 等待用户按键
        key = cv.waitKey(0) & 0xFF
        
        # 处理按键事件
        if key == ord('q'):  # 按 q 键退出
            break
        elif key == 81 or key == ord('a'):  # 左箭头或 a 键，上一张图片
            current_index = (current_index - 1 + total_images) % total_images
        elif key == 83 or key == ord('d'):  # 右箭头或 d 键，下一张图片
            current_index = (current_index + 1) % total_images
    
    # 关闭所有窗口
    cv.destroyAllWindows()

if __name__ == "__main__":
    # 输入包含图像和标注文件的文件夹路径
    image_folder = '/home/hero/Datasets/datasets_from_yuque/origin_none_bal/newnew_bal/B3'
    
    # 调用可视化函数
    visualize_annotations(image_folder)
