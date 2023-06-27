import os
import shutil
import re
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image

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


def log_handle(epoch, train_loss, best_loss, model, log_dir, lr, train_loader, img_latent_dim):
    LOG_FORM = "Epoch: {:02d}, Train Loss: {:.8f}, "
    log = LOG_FORM.format(epoch, train_loss)
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), os.path.join(log_dir, MODEL_DIR_NAME, "best_model.pth"))
        log = log + "save best model"
    if epoch % 20 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, MODEL_DIR_NAME, str(epoch) + "_" + "model.pth"))

    with open(os.path.join(log_dir, LOSS_DIR_NAME, "loss.log"), "a") as f:
        f.write(log + "\n")

    data, target = next(iter(train_loader))
    img, sp = data.to(DEVICE), target.to(DEVICE)
    model.eval()
    img_path = os.path.join(log_dir, TMP_DIR_NAME, f"{epoch}.png")

    # noise = torch.randn(img.size(0), img_latent_dim).to(DEVICE)
    with torch.no_grad():
        g_img = model.only_decode(sp)
        images = torch.cat([img, g_img], dim=0)
        save_image(images, img_path, normalize=True)

    print(log + str(lr))

    return best_loss


def save_loss_pic(log_dir):
    LOG_DRAW = r"Epoch: (\d+), Train Loss: ([0-9.]+)"

    log_file = os.path.join(log_dir, LOSS_DIR_NAME, "loss.log")

    with open(log_file, "r") as f:
        content = f.read()

    epochs = re.findall(LOG_DRAW, content)
    epochs = [(int(epoch), float(train_loss)) for epoch, train_loss in epochs]
    epoch_list, train_loss_list = zip(*epochs)

    plt.plot(epoch_list, train_loss_list, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    pic_dir = os.path.join(log_dir, LOSS_DIR_NAME, "loss.png")
    plt.savefig(pic_dir)


if __name__ == "__main__":
    save_loss_pic(os.path.join("log", "%ex0"))
