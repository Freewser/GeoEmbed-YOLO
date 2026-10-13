import os
import random
import shutil


def split_dataset(images_folder, labels_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, output_folder='split_dataset'):
    """
    将给定的图像和标签文件数据集按照指定比例划分为训练集、验证集和测试集

    :param images_folder: 存放图像文件的文件夹路径
    :param labels_folder: 存放标签文件的文件夹路径
    :param train_ratio: 训练集所占比例，默认值为0.8
    :param val_ratio: 验证集所占比例，默认值为0.1
    :param test_ratio: 测试集所占比例，默认值为0.1
    :param output_folder: 划分后数据集存放的根文件夹路径，默认值为'split_dataset'
    """
    # 检查比例之和是否为1
    assert train_ratio + val_ratio + test_ratio == 1, "比例之和必须为1"

    # 获取图像文件列表
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    random.shuffle(image_files)

    # 计算各数据集的文件数量
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    # 创建输出文件夹及各子集文件夹
    os.makedirs(output_folder, exist_ok=True)
    train_images_folder = os.path.join(output_folder, 'train', 'images')
    train_labels_folder = os.path.join(output_folder, 'train', 'labels')
    val_images_folder = os.path.join(output_folder, 'val', 'images')
    val_labels_folder = os.path.join(output_folder, 'val', 'labels')
    test_images_folder = os.path.join(output_folder, 'test', 'images')
    test_labels_folder = os.path.join(output_folder, 'test', 'labels')
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(test_labels_folder, exist_ok=True)

    # 划分数据集
    for index, file in enumerate(image_files):
        file_name, file_extension = os.path.splitext(file)
        label_file = file_name + '.txt'

        origin_image_path = os.path.join(images_folder, file)
        origin_label_path = os.path.join(labels_folder, label_file)

        if index < train_count:
            target_image_folder = train_images_folder
            target_label_folder = train_labels_folder
        elif index < train_count + val_count:
            target_image_folder = val_images_folder
            target_label_folder = val_labels_folder
        else:
            target_image_folder = test_images_folder
            target_label_folder = test_labels_folder

        shutil.copy2(origin_image_path, target_image_folder)
        shutil.copy2(origin_label_path, target_label_folder)


if __name__ == "__main__":
    # 示例用法，你需要根据实际情况修改下面的文件夹路径
    images_folder_path = 'your_images_folder_path'
    labels_folder_path = 'your_labels_folder_path'
    split_dataset(images_folder_path, labels_folder_path)