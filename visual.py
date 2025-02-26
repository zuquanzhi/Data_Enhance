import cv2 as cv
import os
import numpy as np
from collections import deque

def load_image_and_annotations(image_path, label_path):
    """加载图像和标注"""
    img = cv.imread(image_path)
    if img is None:
        return None, None
    
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            annotations = f.readlines()
    
    return img, annotations

def draw_annotations(img, annotations):
    """在图像上绘制标注"""
    img_copy = img.copy()
    
    for line in annotations:
        parts = line.strip().split()
        class_id = parts[0]
        coords = [float(x) for x in parts[1:]]
        
        points = [(int(coords[i] * img.shape[1]), int(coords[i+1] * img.shape[0])) 
                 for i in range(0, len(coords), 2)]
        
        for i in range(0, len(points), 4):
            quadrilateral = points[i:i+4]
            
            for j in range(len(quadrilateral) - 1):
                cv.line(img_copy, quadrilateral[j], quadrilateral[j+1], 
                       color=(0, 255, 0), thickness=1)
            
            if len(quadrilateral) == 4:
                cv.line(img_copy, quadrilateral[-1], quadrilateral[0], 
                       color=(0, 255, 0), thickness=1)
        
        for point in points:
            cv.circle(img_copy, point, 3, (255, 0, 0), -1)
        
        if len(points) >= 4:
            center = np.mean(points[:4], axis=0).astype(int)
            label_position = (center[0] + 10, center[1] - 10)
            cv.putText(img_copy, class_id, label_position, 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    
    return img_copy

def visualize_annotations(image_folder):
    """
    遍历文件夹内的所有图像和同名的标注文件，显示标注点并连接。
    支持方向键切换图片，按 q 退出。
    """
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()
    
    if not image_files:
        print("文件夹中没有找到任何图片文件。")
        return
    
    # 创建固定窗口
    window_title = "Image Viewer"
    cv.namedWindow(window_title, cv.WINDOW_NORMAL)
    
    # 初始化图像缓存
    cache_size = 3  # 缓存前后各1张图片
    image_cache = {}
    current_index = 0
    total_images = len(image_files)
    
    def preload_images(center_idx):
        """预加载周围的图像"""
        indices = [(center_idx + i) % total_images for i in range(-1, 2)]
        for idx in indices:
            if idx not in image_cache:
                image_file = image_files[idx]
                image_path = os.path.join(image_folder, image_file)
                label_path = os.path.join(image_folder, os.path.splitext(image_file)[0] + ".txt")
                img, annot = load_image_and_annotations(image_path, label_path)
                if img is not None:
                    image_cache[idx] = (img, annot, image_file)
        
        # 清理不需要的缓存
        keys = list(image_cache.keys())
        for k in keys:
            if k not in indices:
                del image_cache[k]
    
    # 首次预加载
    preload_images(current_index)
    
    while True:
        if current_index in image_cache:
            img, annotations, image_file = image_cache[current_index]
            
            # 绘制图像和标注
            img_copy = draw_annotations(img, annotations) if annotations else img.copy()
            
            # 添加图片信息
            info_text = f"File: {image_file} ({current_index + 1}/{total_images})"
            cv.putText(img_copy, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                      0.7, (0, 255, 255), 2, cv.LINE_AA)
            
            rel_path = os.path.join("...", image_file)
            path_text = f"Path: {rel_path}"
            cv.putText(img_copy, path_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 
                      0.7, (0, 255, 255), 2, cv.LINE_AA)
            
            # 更新窗口
            cv.setWindowTitle(window_title, f"Image Viewer - {image_file}")
            cv.imshow(window_title, img_copy)
        
        # 等待用户输入
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 81 or key == ord('a'):
            current_index = (current_index - 1) % total_images
            preload_images(current_index)
        elif key == 83 or key == ord('d'):
            current_index = (current_index + 1) % total_images
            preload_images(current_index)
        elif key == 255:  # 没有按键
            continue
    
    cv.destroyWindow(window_title)

if __name__ == "__main__":
    image_folder = '/home/hero/Datasets_of_Car/TOTAL/4_5_bp/hr'
    visualize_annotations(image_folder)
