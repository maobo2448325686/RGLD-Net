from PIL import Image
import os

# 设置源图片文件夹路径和目标文件夹路径
source_folder = r'D:\Yjs_learning\文章 建筑物屋顶提取\数据集\Massachusetts\png\test_labels'  # 源图片文件夹
target_folder = r'D:\Yjs_learning\文章 建筑物屋顶提取\数据集\Mass512\test\label'  # 目标图片文件夹

# 创建目标文件夹（如果不存在）
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 检查文件是否是图片格式（可以根据需要扩展支持更多格式）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 打开图片
        image_path = os.path.join(source_folder, filename)
        img = Image.open(image_path)

        # 检查图片尺寸是否为1500x1500
        if img.size != (1500, 1500):
            print(f"跳过文件 {filename}，尺寸不符合要求。")
            continue

        # 创建黑色背景画布，确保图片可以被完整切割
        new_img = Image.new('RGB', (1536, 1536), (0, 0, 0))  # 创建黑色背景画布
        new_img.paste(img, (16, 16))  # 将原图片粘贴到画布的中心位置

        # 切割图片
        crop_size = 512
        for row in range(3):  # 6行
            for col in range(3):  # 6列
                # 计算切割区域
                left = col * crop_size
                upper = row * crop_size
                right = left + crop_size
                lower = upper + crop_size

                # 切割图片
                cropped_img = new_img.crop((left, upper, right, lower))

                # 保存切割后的图片
                base_name, extension = os.path.splitext(filename)
                new_filename = f"{base_name}_{row}_{col}{extension}"
                cropped_img.save(os.path.join(target_folder, new_filename))

print("图片切割完成，已保存到目标文件夹。")