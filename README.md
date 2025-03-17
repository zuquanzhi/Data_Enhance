# 数据增强工具集 (Four Points Image Processing Tool)

基于四点坐标的图像处理工具，专门用于目标检测数据集的增强和处理。主要特点是直接使用YOLO格式的四点标注数据进行图像处理，不进行额外的区域扩展。

## 功能特点

- 基于原始四点坐标进行处理
- 保持目标区域的精确性
- 支持多种图像处理模式
- 保留原始标注信息
- 支持批量处理

## 环境要求

- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- tqdm

### 安装依赖

```bash
pip install opencv-python numpy tqdm
```

## 目录结构

```
.
├── four_points_enhance.py   # 主程序文件
├── output/                  # 输出目录
├── images/                  # 测试图像目录
└── README.md               # 说明文档
```

## 使用方法

### 命令行参数

```bash
python four_points_enhance.py \
    --input ./images \
    --output ./output \
    --mode random \
    --start-index 0
```

### 参数说明

- `--input`: 输入图像和标注文件的目录
- `--output`: 处理后图像和标注的保存目录
- `--mode`: 处理模式，可选 ['normal', 'random']
- `--start-index`: 输出文件的起始编号（默认为0）

## 数据格式要求

### 输入图像

- 支持格式: jpg, jpeg, png
- 建议分辨率: > 640x640

### 标注格式 (YOLO格式)

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

- 所有坐标值都是相对值 (0-1)
- 按顺时针或逆时针顺序排列四个点
- 每行一个目标

## 处理流程

1. 读取图像和对应的标注文件
2. 解析四点坐标数据
3. 根据四点坐标进行图像处理
4. 保持原始标注信息不变
5. 保存处理后的图像和标注

## 示例

### 基本使用

```bash
# 处理单个文件
python four_points_enhance.py \
    --input ./images/test.jpg \
    --output ./output

# 处理整个目录
python four_points_enhance.py \
    --input ./images \
    --output ./output
```

### 高级用法

```bash
# 使用随机模式
python four_points_enhance.py \
    --input ./images \
    --output ./output \
    --mode random

# 指定起始索引
python four_points_enhance.py \
    --input ./images \
    --output ./output \
    --start-index 100
```

## 注意事项

1. 确保输入图像和标注文件同名（仅扩展名不同）
2. 处理前备份原始数据
3. 检查输出结果的正确性
4. 标注坐标需要按顺序排列

## 常见问题

1. **标注文件找不到**
   - 检查标注文件名是否与图像文件名相同
   - 确认标注文件扩展名为 .txt

2. **坐标解析错误**
   - 确认标注格式是否正确
   - 检查坐标值是否在0-1范围内

3. **处理结果不理想**
   - 确认输入图像质量
   - 检查标注点的顺序是否正确

## 最佳实践

1. 先在小数据集上测试
2. 定期检查处理结果
3. 保持标注格式的一致性
4. 使用有意义的文件命名

## 开发计划

- [ ] 添加更多处理模式
- [ ] 支持自定义处理参数
- [ ] 添加数据增强功能
- [ ] 优化处理性能

## 许可证

MIT License

## 贡献指南

1. Fork 项目
2. 创建新分支
3. 提交更改
4. 发起 Pull Request

## 作者

[作者名称]

## 更新日志

### v1.0.0 (2024-03-17)
- 初始版本发布
- 基础功能实现