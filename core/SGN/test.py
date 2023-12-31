import os.path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from core.dt.absorber.loader_build import Dataloader_BoxCox, Dataset_BoxCox
from model import sgn
from core.SPN.model import spn
import util

DEVICE = util.DEVICE

log_dir = os.path.join("log", "%ex0")

dloader = Dataloader_BoxCox("cross_data")
train_loader, val_loader, test_loader = dloader.get()
mt_data = dloader.get_metadata()
scale = mt_data[2]
y_len = mt_data[3]

model, img_latent_dim = sgn.get_sgn()
model = model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(log_dir, util.MODEL_DIR_NAME, "best_model.pth")))
model.eval()

SPN, num_class = spn.get_spn()
SPN = SPN.to(DEVICE)
SPN.load_state_dict(torch.load(os.path.join("ext", "SPN.pth")))
SPN.eval()

count = 0

for data, target in train_loader:
    imgs, sps = data.to(DEVICE).detach(), target.to(DEVICE).detach()
    b_size = imgs.size(0)

    g_imgs = model.only_decode(sps)
    g_imgs[g_imgs >= 0.5] = 1.0
    g_imgs[g_imgs < 0.5] = 0.0
    g_sps = SPN(g_imgs)

    test_loss = torch.nn.functional.mse_loss(g_sps, sps)
    TEST_LOSS_FORM = "Test Loss: {:.8f}, "
    test_loss_log = TEST_LOSS_FORM.format(test_loss)
    print(test_loss_log)
    with open(os.path.join(log_dir, util.TEST_DIR_NAME, "test_loss.log"), "a") as f:
        f.write(test_loss_log + "\n")

    sps = sps.cpu().detach().numpy()
    g_sps = g_sps.cpu().detach().numpy()
    imgs = imgs.cpu().numpy()
    g_imgs = g_imgs.detach().cpu().numpy()

    for i in range(b_size):
        count += 1

        img = imgs[i][0]
        g_img = g_imgs[i][0]
        sp = sps[i]
        g_sp = g_sps[i]

        os.makedirs(os.path.join(log_dir, util.TEST_DIR_NAME, str(count)))
        util.save_mat_rat90(img, os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "mat.png"))
        util.save_mat_rat90(g_img, os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "g_mat.png"))
        util.save_mat_data(img, os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "mat.txt"))
        util.save_mat_data(g_img, os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "g_mat.txt"))

        np.savetxt(os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "sp_bc.txt"), sp)
        np.savetxt(os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "g_sp_bc.txt"), g_sp)

        plt.plot(np.linspace(1, 5, int(y_len)), sp, label="Simulation")
        plt.plot(np.linspace(1, 5, int(y_len)), g_sp, label="Predict")
        plt.ylim(-scale / 2, scale / 2)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Absorptivity(Box-Cox)')
        plt.legend()
        plt.savefig(os.path.join(log_dir, util.TEST_DIR_NAME, str(count) + "_sp_bc.png"))
        plt.close()

        # images = torch.cat([img.unsqueeze(0), g_img.unsqueeze(0)], dim=0)
        # save_image(images, os.path.join("./", log_dir, util.TEST_DIR_NAME, str(count) + "_img.png"), normalize=True)

        sp = dloader.rev_resolution(np.expand_dims(sp, axis=0))[0]
        g_sp = dloader.rev_resolution(np.expand_dims(g_sp, axis=0))[0]

        np.savetxt(os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "sp.txt"), sp)
        np.savetxt(os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "g_sp.txt"), g_sp)

        # 绘制曲线图
        plt.plot(np.linspace(1, 5, int(y_len)), sp, label="Simulation")
        plt.plot(np.linspace(1, 5, int(y_len)), g_sp, label="Predict")
        plt.ylim(0, 1)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Absorptivity')
        plt.legend()
        plt.savefig(os.path.join(log_dir, util.TEST_DIR_NAME, str(count) + "_sp.png"))
        plt.close()
