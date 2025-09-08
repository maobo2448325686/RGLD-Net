from PIL import Image
import os

input_folder = r'D:\maobo\maobo_research\zhaihai\code\MyNet\BENet\data\guiyang512\Test\image'   # 输入文件夹路径
output_folder = r'D:\maobo\maobo_research\zhaihai\code\MyNet\BENet\data\guiyang256\test\image'   # 输出文件夹路径
crop_size = 256  # 裁剪后的尺寸

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹内所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 打开图片
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            # 对图片进行裁剪
            w, h = img.size
            row_num = 1  # 行号
            for i in range(0, h-crop_size+1, crop_size):
                col_num = 1  # 列号
                for j in range(0, w-crop_size+1, crop_size):
                    box = (j, i, j+crop_size, i+crop_size)
                    crop = img.crop(box)
                    # 生成文件名并保存裁剪后的小块到输出文件夹
                    output_filename = f'{filename.split(".")[0]}_{row_num}_{col_num}.png'
                    output_path = os.path.join(output_folder, output_filename)
                    crop.save(output_path)
                    col_num += 1
                row_num += 1
