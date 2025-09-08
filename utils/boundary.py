import cv2
import numpy as np
import os
import glob

# 定义输入和输出文件夹路径
label_folder = r"E:\mb\jianzhuwutiqu\data\demo256\val\label"  # 替换为你的标签文件夹路径
boundary_folder = r"E:\mb\jianzhuwutiqu\data\demo256\val\boundary_png" # 替换为你想要保存边界图像的文件夹路径

# 如果输出文件夹不存在，则创建它
if not os.path.exists(boundary_folder):
    os.makedirs(boundary_folder)

# 获取所有标签图像的路径
label_image_paths = glob.glob(os.path.join(label_folder, '*.png'))

# 遍历每个标签图像
for label_image_path in label_image_paths:
    # 读取标签图像
    label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)

    # 确保标签图像是二值图像（0 和 255）
    _, binary_label = cv2.threshold(label_image, 128, 255, cv2.THRESH_BINARY)

    # 应用 Canny 算子生成边界标签
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(binary_label, low_threshold, high_threshold)

    # 保存边界图像
    # 获取原始图像的文件名
    filename = os.path.basename(label_image_path)
    boundary_image_path = os.path.join(boundary_folder, filename)

    # 保存边界图像
    cv2.imwrite(boundary_image_path, edges)

# print(f"所有边界图像已保存到 {boundary_folder}")