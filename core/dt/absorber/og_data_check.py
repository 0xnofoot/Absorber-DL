import os.path
import numpy as np

from core import global_var


# ----------------------#
# 对仿真的原始数据进行预处理
# 使数据能够被 Pytorch 快速读取

# 单层金属吸收器的数据预处理函数，将仿真的数据处理成 npy 格式
# 方便构造 Pytorch 数据集
# 在该函数中存在几处校验数据的代码,实际上应该是不需要的
# 但是为了安全考虑依然选择校验
# index 是数据索引的开始值,一般给 0 ，如果要添加数据集手动给值
# 使用前必须手动在对应的数据文件夹下建好 check 文件夹，再在 check 下 建好 x 和 y 文件夹
# 例子：Project_Root -> data -> original -> absorber -> check -> (x, y)
# 数据集生产后手动转移到需要的地方，并更改好文件夹名
def absorber(index=0, mutil_project_dir="absorber"):
    base_dir = os.path.join(global_var.data_dir, "original", mutil_project_dir)
    mat_data_dir = os.path.join("data", "matrix", "data", "layer_1")
    sParam_data_dir = os.path.join("data", "cst", "sParam", "Zmax1_Zmax1")
    check_dir = os.path.join(base_dir, "check")

    x_buff = []
    y_buff = []

    def x_handle(x_file, index):
        x = np.loadtxt(x_file)
        if x.shape == (64, 64) and x.min() == 0 and x.max() == 1:
            x_data = np.loadtxt(x_file)

            x_buff.append(x_data)
            if x_buff[index].all() != x_data.all():
                print("x 数据添加存在问题")
                print("index: " + str(index))

    def y_handle(y_file, index):
        # 奇异点检测, 丢弃奇异点数据
        try:
            y_data = np.loadtxt(y_file)
        except ValueError:
            y_name = os.path.basename(y_file)
            return y_name

        with open(y_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) != 1001:
                y_name = os.path.basename(y_file)
                return y_name

        # 转置矩阵，实现行列互换
        y_data = np.transpose(y_data)
        # # 删除第一列频率值数据，得到 2x1001 的 numpy 数组,分别是 S11 的实部和虚部
        y_data = np.delete(y_data, 0, axis=0)
        # 计算新的一列吸收率数据，
        new_column = 1 - (y_data[0] ** 2 + y_data[1] ** 2)

        # 吸收率检测，丢弃没有吸收效果的数据,和一些因为仿真异常造成的最小值小于等于 0 的数据
        max_value = np.max(new_column)
        min_value = np.min(new_column)
        if max_value < 0.5 or max_value >= 1.0 or min_value < 0.001:
            y_name = os.path.basename(y_file)
            return y_name

        # 添加到数组中，得到 3x1001 的 numpy 数组
        y_data = np.vstack((y_data, new_column))
        # 归一化第二列和第三列数据，将范围从 -1 到 1 变为 0 到 1
        y_data[0] = (y_data[0] + 1) / 2
        y_data[1] = (y_data[1] + 1) / 2

        y_buff.append(y_data)
        if y_buff[index].all() != y_data.all():
            print("y 数据添加存在问题")
            print("index: " + str(index))

    # 获取单个数据所在路径的 list
    # 并做了一个 矩阵数据 和 S 参数 数据是否大小一致的校验
    # project_dir: 数据文件夹中每一个工程文件夹名
    def get_data_path_list(project_dir):

        x_path_list = os.listdir(os.path.join(project_dir, mat_data_dir))
        y_path_list = os.listdir(os.path.join(project_dir, sParam_data_dir))

        if len(x_path_list) != len(y_path_list):
            print(project_dir + "数据不匹配")
            return

        dpl = []
        for i in range(len(x_path_list)):
            dpl.append((x_path_list[i], y_path_list[i]))

        return dpl

    info = {}

    # info 处理的函数
    # 在 info 字典中记录这次数据集构建的一些信息
    def info_handle(project_dir, drop_l, index_contrast):
        index_file = os.path.join(project_dir, "index", "index.txt")
        project_name = os.path.basename(project_dir)
        info[project_name] = {}
        with open(index_file, "r") as f:
            lines = f.readlines()
            index_start = lines[0].split("\t")[0]
            index_stop = lines[-1].split("\t")[0]
            info[project_name]["o_index_range"] = index_start + "~" + index_stop

        info[project_name]["o_index_drop"] = drop_l
        info[project_name]["index_contrast"] = index_contrast

    project_list = os.listdir(base_dir)

    # 对文件夹中每个单独的工程文件夹处理
    for project_name in project_list:
        if project_name == "check" or project_name == "moment":
            continue
        project_dir = os.path.join(base_dir, project_name)

        data_path_list = get_data_path_list(project_dir)

        drop_list = []
        index_contrast = []

        for data_path in data_path_list:
            x_file = data_path[0]
            y_file = data_path[1]

            if x_file.split("_")[0] != y_file.split("_")[0]:
                print(project_dir + "数据不匹配---" + x_file + "---" + y_file)
                return

            o_index = x_file.split("_")[0]

            y_file = os.path.join(project_dir, sParam_data_dir, y_file)
            # 奇异点以及无吸收特征检测
            y_name = y_handle(y_file, index)
            if y_name is not None:
                drop_list.append(y_name.split("_")[0])
                continue

            x_file = os.path.join(project_dir, mat_data_dir, x_file)
            x_handle(x_file, index)

            index_contrast.append(str(o_index) + "--" + str(index))

            index += 1

        info_handle(project_dir, drop_list, index_contrast)

    x_buff = np.array(x_buff)
    y_buff = np.array(y_buff)
    np.save(os.path.join(check_dir, "x.npy"), x_buff)
    np.save(os.path.join(check_dir, "y.npy"), y_buff)

    with open(os.path.join(check_dir, "info.txt"), "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
            f.write("\n")


# 调用以构建 npy 数据集
if __name__ == "__main__":
    absorber(mutil_project_dir="cross_data")
