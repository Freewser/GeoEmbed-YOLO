import os
import random
import shutil

# 原始图像数据文件夹
image_dir = "E:/data/slab/pre-data/images"
# 原始标注文件文件夹
label_dir = "E:/data/slab/pre-data/labels - hard"
# 划分后训练集图像文件夹
train_image_dir = "E:/data/slab/real_data-1/images/train"
# 划分后训练集标注文件夹
train_label_dir = "E:/data/slab/real_data-1/labels/train"
# 划分后验证集图像文件夹
val_image_dir = "E:/data/slab/real_data-1/images/val"
# 划分后验证集标注文件夹
val_label_dir = "E:/data/slab/real_data-1/labels/val"

# 划分比例，这里假设80%的数据用于训练，20%用于验证
split_ratio = 0.14

# 创建训练集和验证集相关文件夹
os.makedirs('E:/data/slab/real_data-1/images/train', exist_ok=True)
os.makedirs('E:/data/slab/real_data-1/labels/train', exist_ok=True)
os.makedirs('E:/data/slab/real_data-1/images/val', exist_ok=True)
os.makedirs('E:/data/slab/real_data-1/labels/val', exist_ok=True)

# 获取所有图像文件名（此时图像文件后缀是.png）
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
random.shuffle(image_files)

# 计算划分的索引位置
split_index = int(len(image_files) * split_ratio)

# 划分训练集和验证集的图像及标注文件
train_images = image_files[:split_index]
val_images = image_files[split_index:]

for img_file in train_images:
    # 复制图像文件到训练集图像文件夹
    shutil.copy(os.path.join(image_dir, img_file), os.path.join(train_image_dir, img_file))
    # 获取对应的标注文件名（标注文件名和图像文件名除后缀外一致，只是后缀为.txt）
    label_file = img_file.replace('.png', '.txt')
    # 复制标注文件到训练集标注文件夹
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))

for img_file in val_images:
    # 复制图像文件到验证集图像文件夹
    shutil.copy(os.path.join(image_dir, img_file), os.path.join(val_image_dir, img_file))
    # 获取对应的标注文件名（标注文件名和图像文件名除后缀外一致，只是后缀为.txt）
    label_file = img_file.replace('.png', '.txt')
    # 复制标注文件到验证集标注文件夹
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))

print("数据集划分完成！")

# 以下是简单模拟训练过程（只是打印信息，实际要结合真实模型等进行操作）
epochs = 10
for epoch in range(epochs):
    print(f"开始第 {epoch + 1} 轮训练...")
    # 这里假设加载训练集数据进行训练，实际要通过数据加载器等方式处理
    train_data = os.listdir(train_image_dir)
    for data in train_data:
        print(f"正在处理训练数据: {data}")
    print(f"第 {epoch + 1} 轮训练完成，开始验证...")
    # 这里假设加载验证集数据进行验证，实际要通过数据加载器等方式处理
    val_data = os.listdir(val_image_dir)
    for data in val_data:
        print(f"正在处理验证数据: {data}")
    print(f"第 {epoch + 1} 轮验证完成")