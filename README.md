# 图像增强工具集 (Image Enhance Tools)

用于图像处理和数据增强的 Python 工具集合，专注于 YOLO 格式目标检测数据集的处理和增强。

## 环境要求

- Python 3.8+
- OpenCV
- NumPy
- tqdm

### 安装依赖

```bash
# 创建新的 conda 环境
conda create -n imgtools python=3.8
conda activate imgtools

# 安装依赖包
conda install -c conda-forge opencv numpy tqdm
```

## 工具说明

### 1. 亮度调整工具 (lightness.py)

调整图像的亮度和对比度，支持全图调整和区域调整。

```bash
# 基本使用
python lightness.py --input images/ --output output/ --alpha 1.6 --beta 25

# 仅处理标注区域内部
python lightness.py --input images/ --output output/ --mode inside --alpha 1.8 --beta 20

# 参数说明:
# --alpha: 对比度因子 (默认: 1.6)
# --beta: 亮度因子 (默认: 25)
# --mode: 处理模式 [full_image/inside/outside/random/superrandom] (默认: full_image)
```

### 2. 噪声添加工具 (noise.py)

为图像添加随机噪声、阴影和锐化效果。

```bash
# 添加所有效果
python noise.py --input images/ --output output/ --noise --shadow --sharpen

# 仅添加噪声
python noise.py --input images/ --output output/ --noise

# 参数说明:
# --noise: 添加高斯噪声
# --shadow: 添加随机阴影
# --sharpen: 应用锐化效果
```

### 3. HSV 亮度调整工具 (low_hsv.py)

针对装甲板区域的亮度调整，支持自适应区域扩展。

```bash
# 基本使用
python low_hsv.py --input images/ --output output/

# 指定缩放因子
python low_hsv.py --input images/ --output output/ 

```

### 4. 图像替换工具 (replace.py)

支持图像区域的替换和透视变换，用于数据增强。

```bash
# 基本使用
python replace.py --src source/ --dst target/ --output output/ --start 100 --min_cat 23

# 参数说明:
# --src: 源图像目录
# --dst: 目标图像目录
# --start: 起始索引 (默认: 100)
# --min_cat: 最小类别ID (默认: 23)
```

### 5. 视频处理工具 (video_to_images.py)

将视频转换为连续帧图像。

```bash
# 基本使用
python video_to_images.py --video input.mp4 --output frames/

# 指定帧率
python video_to_images.py --video input.mp4 --output frames/ --fps 30

# 参数说明:
# --fps: 提取帧率 (默认: 原视频帧率)
```

### 6. 标注可视化工具 (visual.py)

可视化 YOLO 格式的标注文件。

```bash
python visual.py --image image
```

## 文件格式要求

### YOLO 标注格式
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

示例:
```
0 100 200 150 200 150 250 100 250
```

## 常见问题

1. **ImportError: No module named cv2**
   ```bash
   conda install -c conda-forge opencv
   ```

2. **Permission denied**
   ```bash
   chmod +x *.py
   ```

3. **标签文件未找到**
   - 确保标签文件与图像文件同名，仅扩展名不同
   - 图像文件扩展名支持: .jpg, .jpeg, .png, .bmp
   - 标签文件必须为 .txt 格式

## 最佳实践

1. 处理前备份原始数据
2. 使用虚拟环境管理依赖
3. 先在小数据集上测试
4. 检查输出结果的正确性

## 目录结构

```
Tools/
├── lightness.py     # 亮度调整工具
├── noise.py         # 噪声添加工具
├── low_hsv.py       # HSV亮度调整工具
├── replace.py       # 图像替换工具
├── video_to_images.py   # 视频处理工具
├── visual.py        # 标注可视化工具
└── README.md        # 说明文档
```

## 许可证

MIT License