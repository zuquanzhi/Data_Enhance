import cv2 as cv
import os
import numpy as np

def visualize_annotations(image_path, label_path):
    """
    读取图像和同名的标注文件，显示标注点并连接。
    :param image_path: 图像文件路径
    :param label_path: 标注文件路径
    :return: 处理后的图像
    """
    # 读取图像
    img = cv.imread(image_path)
    
    if img is None:
        print(f"无法加载图像: {image_path}")
        return None
    
    # 读取标注文件
    if not os.path.exists(label_path):
        print(f"标注文件不存在: {label_path}")
        return None
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = parts[0]
            coords = [float(x) for x in parts[1:]]
            points = [(int(coords[i] * img.shape[1]), int(coords[i+1] * img.shape[0])) 
                     for i in range(0, len(coords), 2)]
            
            for i in range(0, len(points), 4):
                quadrilateral = points[i:i+4]
                for j in range(len(quadrilateral) - 1):
                    cv.line(img, quadrilateral[j], quadrilateral[j+1], 
                           color=(0, 255, 0), thickness=1)
                if len(quadrilateral) == 4:
                    cv.line(img, quadrilateral[-1], quadrilateral[0], 
                           color=(0, 255, 0), thickness=1)
            
            for point in points:
                cv.circle(img, point, 3, (255, 0, 0), -1)
            
            if len(points) >= 4:
                center = np.mean(points[:4], axis=0).astype(int)
                label_position = (center[0] + 10, center[1] - 10)
                cv.putText(img, class_id, label_position, 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    
    return img

def visualize_folder(folder_path):
    """
    可视化文件夹中的所有图像及其标注。
    :param folder_path: 包含图像和标注文件的文件夹路径
    """
    # 获取所有图像文件
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"在 {folder_path} 中未找到图像文件")
        return
    
    current_index = 0
    
    while True:
        image_file = image_files[current_index]
        image_path = os.path.join(folder_path, image_file)
        label_path = os.path.join(folder_path, 
                                 os.path.splitext(image_file)[0] + '.txt')
        
        img = visualize_annotations(image_path, label_path)
        if img is not None:
            # 显示当前图像信息
            info = f"{image_file} ({current_index + 1}/{len(image_files)})"
            cv.putText(img, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv.imshow("Image with annotations", img)
            
            # 等待键盘输入
            key = cv.waitKey(0) & 0xFF
            
            if key == ord('q'):  # 按 q 退出
                break
            elif key == 83 or key == ord('d'):  # 右箭头或 'd'
                current_index = (current_index + 1) % len(image_files)
            elif key == 81 or key == ord('a'):  # 左箭头或 'a'
                current_index = (current_index - 1) % len(image_files)
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    # 指定包含图像和标注文件的文件夹路径
    folder_path = './output'
    
    # 调用文件夹可视化函数
    visualize_folder(folder_path)