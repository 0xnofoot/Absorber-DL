import os
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from core import global_var


# 构建对于单层金属吸收器的 CST 仿真数据的 Dataset 类
class Dataset_BoxCox(data.Dataset):
    def __init__(self, check_data_dir, left, right, scale, y_len):
        if left >= right:
            print("left左边界必须小于right右边界！！")
            return
        if scale <= 0:
            print("scale缩放尺度必须大于0！！")
            return

        x_data = np.load(os.path.join(check_data_dir, "x.npy"))
        y_data = np.load(os.path.join(check_data_dir, "y.npy"))

        if x_data.shape[0] != y_data.shape[0]:
            print("数据长度不一致-1!!!")
            print("请检查原始数据构造代码！！！")
            return

        y_data = y_data[:, 2:, :-1]
        y_data = y_data[:, :, ::int(1000 / y_len)]
        y_data = np.squeeze(y_data)

        s = y_data.shape[0]
        y_data = y_data.reshape(-1)
        buff, lambda_ = scipy.stats.boxcox(y_data)

        #################
        print("lambda: " + str(lambda_))
        print(f"第一次变换后的数据范围：[{buff.min()}, {buff.max()}]")
        counts, bins = np.histogram(buff, bins=10000, range=(buff.min(), buff.max()))
        # 绘制柱状图
        plt.bar(bins[:-1], counts, width=0.01)
        plt.show()
        plt.close()
        # data_inv = scipy.special.inv_boxcox(buff.reshape(-1), lambda_)
        #################

        buff = buff.reshape(s, y_len)
        min_vals = np.min(buff, axis=1)
        max_vals = np.max(buff, axis=1)
        delete_cols = np.where(np.logical_or(min_vals <= left, max_vals >= right))[0]
        buff = np.delete(buff, delete_cols, axis=0)

        #################
        print(f"第二次变换后的数据范围：[{buff.min()}, {buff.max()}]")
        counts, bins = np.histogram(buff, bins=10000, range=(buff.min(), buff.max()))
        plt.bar(bins[:-1], counts, width=0.01)
        plt.show()
        plt.close()
        #################

        buff = buff * scale / (right - left) + np.abs(right * scale / (right - left)) + scale / 2

        #################
        print(f"第三次变换后的数据范围：[{buff.min()}, {buff.max()}]")
        counts, bins = np.histogram(buff, bins=10000, range=(buff.min(), buff.max()))
        plt.bar(bins[:-1], counts, width=0.01)
        plt.show()
        plt.close()
        #################

        y_data = buff
        x_data = np.delete(x_data, delete_cols, axis=0)

        if x_data.shape[0] != y_data.shape[0]:
            print("数据长度不一致-2!!!")
            print("请检查该函数！！！")
            return

        self.x = torch.from_numpy(x_data).type(torch.float32)
        self.x = torch.unsqueeze(self.x, 1)
        self.y = torch.from_numpy(y_data).type(torch.float32)

        self.lambda_ = lambda_
        self.num_samples = x_data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


# 获取 dataloader 的函数, boxcox 变换后的数据
# check_dir_name: 数据集源数据所在文件夹
# 所有的数据应该都放在工程目录下的 data 文件夹中
# 该文件夹有两个子文件夹，分别是 original 和 check
# original 用于保存原始数据，比如从 CST 仿真得到的矩阵数据和 S 参数
# 原始数据经过处理成 .npy 格式的数据，保存在 check 文件夹中
# 原始数据的处理函数单独写脚本处理，因为对于不同的工程产生的数据不同，数据处理的目的也不同
# check 文件夹中的数据交由 torch 进行处理构建 dataset
# 注意的是，构建 dataset 时需要指定数据源所在的文件夹，即我要在 check 目录的哪个文件夹中读取数据
# 该文件夹必须包含 x 和 y 两个文件夹，分别代表 features 和 targets
# 例子：Project_Root -> data -> check -> absorber -> (x, y)
class Dataloader_BoxCox:
    def __init__(self, loader_dir_name):
        self.loader_dir = os.path.join(global_var.data_dir, "loader", loader_dir_name)

    def get_metadata(self):
        metadata_p = os.path.join(self.loader_dir, "metadata.txt")
        with open(metadata_p, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            if lines[0] != "Box-Cox":
                print("数据存在错误，该类只能处理Box-Cox变换的数据，请查阅源码")
                return
            left = np.float64(lines[1])
            right = np.float64(lines[2])
            scale = np.float64(lines[3])
            y_len = np.float64(lines[4])
            lambda_ = np.float64(lines[5])
            metadata = [left, right, scale, y_len, lambda_]
        return metadata

    @staticmethod
    def fetch(loader_dir):
        train_p = os.path.join(loader_dir, "train_loader.pickle")
        val_p = os.path.join(loader_dir, "val_loader.pickle")
        test_p = os.path.join(loader_dir, "test_loader.pickle")
        with open(train_p, "rb") as f:
            train_l = pickle.load(f)
        with open(val_p, "rb") as f:
            val_l = pickle.load(f)
        with open(test_p, "rb") as f:
            test_l = pickle.load(f)
        return train_l, val_l, test_l

    # 构建 loader 和 获取 loader 都是这个函数
    # 获取 loader不需要传参
    # 最好先用这个文件构建 loader，再其他地方无参获取loader
    def get(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, bsize=64, check_dir_name=None, metadata=None):
        loader_dir = self.loader_dir

        if os.path.exists(loader_dir):
            train_l, val_l, test_l = self.fetch(loader_dir)
            return train_l, val_l, test_l

        if check_dir_name is None:
            print("未找到以构建的loader，必须重新构建，请指定源数据所在文件夹名")
            return

        os.makedirs(loader_dir)

        check_dir = os.path.join(global_var.data_dir, "check", check_dir_name)

        dataset = Dataset_BoxCox(check_dir, left=metadata[0], right=metadata[1], scale=metadata[2], y_len=metadata[3])
        metadata[4] = dataset.lambda_

        # 计算数据集的大小
        dataset_size = dataset.num_samples

        # 计算训练集和验证集的大小
        train_size = int(dataset_size * train_ratio)
        test_size = int(dataset_size * test_ratio)
        val_size = dataset_size - train_size - test_size

        # 使用 random_split 方法将数据集分成训练集和验证集
        train_dataset, val_dataset, test_dataset = data.random_split(dataset, [train_size, val_size, test_size])

        # 将训练集和验证集分别封装成 DataLoader 对象
        train_l = data.DataLoader(train_dataset, batch_size=bsize, shuffle=True, drop_last=True, pin_memory=True)
        val_l = data.DataLoader(val_dataset, batch_size=bsize, shuffle=True, drop_last=True, pin_memory=True)
        test_l = data.DataLoader(test_dataset, batch_size=bsize, shuffle=True, drop_last=True, pin_memory=True)

        # 序列化对象并保存到文件中
        train_p = os.path.join(loader_dir, "train_loader.pickle")
        val_p = os.path.join(loader_dir, "val_loader.pickle")
        test_p = os.path.join(loader_dir, "test_loader.pickle")
        metadata_p = os.path.join(loader_dir, "metadata.txt")
        with open(train_p, "wb") as f:
            pickle.dump(train_l, f)
        with open(val_p, "wb") as f:
            pickle.dump(val_l, f)
        with open(test_p, "wb") as f:
            pickle.dump(test_l, f)
        with open(metadata_p, "w") as f:
            # 写入第一行字符串
            f.write('Box-Cox\n')
            # 写入元组数据
            for i in metadata:
                f.write('%.15f\n' % i)

        return train_l, val_l, test_l

    def rev_resolution(self, data):
        metadata = self.get_metadata()

        left = metadata[0]
        right = metadata[1]
        scale = metadata[2]
        y_len = metadata[3]
        lambda_ = metadata[4]

        bsize = data.shape[0]
        yl = data.shape[1]

        if yl != y_len:
            print("输入的数据长度与该loader中的数据长度不匹配，请自行构建反向解析函数")
            return

        if data.min() <= -scale / 2 or data.max() >= scale / 2:
            print("错误0")

        data = (data - scale / 2 - np.abs(right * scale / (right - left))) * (right - left) / scale
        if data.min() <= left or data.max() >= right:
            print("错误1")

        data = scipy.special.inv_boxcox(data.reshape(-1), lambda_)

        if data.min() <= 0 or data.max() >= 1:
            print("错误2")

        data = data.reshape(bsize, yl)

        return data


# 调用以构建 loader
if __name__ == "__main__":
    check_dir_name = "cross_data_new"
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    batch_size = 64
    metadata = [-40, 0, 2, 1000, 0]
    train_loader, val_loader, test_loader = Dataloader_BoxCox("cross_data_new").get(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        bsize=batch_size,
        check_dir_name=check_dir_name,
        metadata=metadata
    )
    pass
