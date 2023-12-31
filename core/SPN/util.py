import os
import re
import shutil
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR_NAME = "test"
TMP_DIR_NAME = "tmp"
MODEL_DIR_NAME = "model"
LOSS_DIR_NAME = "loss"
STATE_DIR_NAME = "state"


def create_log_dir(dir_name=None):
    if dir_name is None:
        current_datetime = datetime.now()
        log_dir = os.path.join("log", current_datetime.strftime("%Y_%m_%d-%H_%M_%S"))
    else:
        log_dir = dir_name

    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, TEST_DIR_NAME))
    os.makedirs(os.path.join(log_dir, TMP_DIR_NAME))
    os.makedirs(os.path.join(log_dir, MODEL_DIR_NAME))
    os.makedirs(os.path.join(log_dir, LOSS_DIR_NAME))
    os.makedirs(os.path.join(log_dir, STATE_DIR_NAME))

    files_to_copy = ["train.py", "test.py", "util.py"]
    for file in files_to_copy:
        shutil.copy(file, os.path.join(log_dir, STATE_DIR_NAME))
    shutil.copytree("model", os.path.join(os.path.join(log_dir, STATE_DIR_NAME), "model"))

    return log_dir


def log_handle(epoch, train_loss, val_loss, best_loss, model, log_dir, lr):
    LOG_FORM = "Epoch: {:02d}, Train Loss: {:.8f}, Val Loss: {:.8f}, "
    log = LOG_FORM.format(epoch, train_loss, val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), os.path.join(log_dir, MODEL_DIR_NAME, "best_model.pth"))
        log = log + "save best model"
    if epoch % 20 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, MODEL_DIR_NAME, str(epoch) + "_" + "model.pth"))

    with open(os.path.join(log_dir, LOSS_DIR_NAME, "loss.log"), "a") as f:
        f.write(log + "\n")

    print(log + " " + str(lr))

    return best_loss


def save_loss_pic(log_dir):
    LOG_DRAW = r"Epoch: (\d+), Train Loss: ([0-9.]+), Val Loss: ([0-9.]+)"

    log_file = os.path.join(log_dir, LOSS_DIR_NAME, "loss.log")

    with open(log_file, "r") as f:
        content = f.read()

    epochs = re.findall(LOG_DRAW, content)
    epochs = [(int(epoch), float(train_loss), float(val_loss)) for epoch, train_loss, val_loss in epochs]
    epoch_list, train_loss_list, val_loss_list = zip(*epochs)

    plt.plot(epoch_list, train_loss_list, label="Training Loss")
    plt.plot(epoch_list, val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    pic_dir = os.path.join(log_dir, LOSS_DIR_NAME, "loss.png")
    plt.savefig(pic_dir)


# 以逆时针旋转 90°显示该矩阵
def save_mat_rat90(matrix, save_path):
    matrix = np.rot90(matrix, 1)
    plt.matshow(matrix, cmap=plt.cm.Greys)
    plt.xticks(alpha=0)
    plt.yticks(alpha=0)
    plt.tick_params(axis='x', width=0)
    plt.tick_params(axis='y', width=0)
    # plt.grid()
    plt.savefig(save_path)
    plt.close("all")


def save_mat_data(matrix, save_path):
    # 矩阵以 %d 数据类型保存，并以" "做分隔符保存到 test.txt文件中
    np.savetxt(save_path, matrix, fmt="%d", delimiter=" ")


if __name__ == "__main__":
    save_loss_pic(os.path.join("log", "2023_06_13-00_12_24"))
