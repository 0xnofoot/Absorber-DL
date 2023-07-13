import os

from matplotlib import pyplot as plt

from core import global_var
import numpy as np
import scipy

check_data_dir = os.path.join(global_var.data_dir, "check", "cross_data")
y_data = np.load(os.path.join(check_data_dir, "y.npy"))

y_len = 1000
scale = 2.0
left = -40
right = 0

y_data = y_data[:, 2:, :-1]
y_data = y_data[:, :, ::int(1000 / y_len)]
y_data = np.squeeze(y_data)

s = y_data.shape[0]
y_data = y_data.reshape(-1)

#################
np.savetxt("0.txt", y_data)
counts, bins = np.histogram(y_data, bins=10000, range=(y_data.min(), y_data.max()))
plt.bar(bins[:-1], counts, width=0.01)
plt.savefig("0.png")
plt.close()
#################

buff, lambda_ = scipy.stats.boxcox(y_data)

#################
np.savetxt("1.txt", buff)
print("lambda: " + str(lambda_))
print(f"第一次变换后的数据范围：[{buff.min()}, {buff.max()}]")
counts, bins = np.histogram(buff, bins=10000, range=(buff.min(), buff.max()))
# 绘制柱状图
plt.bar(bins[:-1], counts, width=0.01)
plt.savefig("1.png")
plt.close()
#################

buff = buff.reshape(s, y_len)
min_vals = np.min(buff, axis=1)
max_vals = np.max(buff, axis=1)
delete_cols = np.where(np.logical_or(min_vals <= left, max_vals >= right))[0]
buff = np.delete(buff, delete_cols, axis=0)

#################
np.savetxt("2.txt", buff.reshape(-1))
print(f"第二次变换后的数据范围：[{buff.min()}, {buff.max()}]")
counts, bins = np.histogram(buff, bins=10000, range=(buff.min(), buff.max()))
plt.bar(bins[:-1], counts, width=0.01)
plt.savefig("2.png")
plt.close()
#################

buff = buff * scale / (right - left) + np.abs(right * scale / (right - left)) + scale / 2

#################
np.savetxt("3.txt", buff.reshape(-1))
print(f"第三次变换后的数据范围：[{buff.min()}, {buff.max()}]")
counts, bins = np.histogram(buff, bins=10000, range=(buff.min(), buff.max()))
plt.bar(bins[:-1], counts, width=0.01)
plt.savefig("3.png")
plt.close()
#################
