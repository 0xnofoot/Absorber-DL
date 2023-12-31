import os.path

import torch
import numpy as np
import matplotlib.pyplot as plt
from core.dt.absorber.loader_build import Dataloader_BoxCox, Dataset_BoxCox
from model import sfen
import util

DEVICE = util.DEVICE

log_dir = os.path.join("log", "%ex0")

dloader = Dataloader_BoxCox("cross_data")
train_loader, val_loader, test_loader = dloader.get()
mt_data = dloader.get_metadata()
scale = mt_data[2]
y_len = mt_data[3]

model, latent_dim = sfen.get_sfen()
model = model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(log_dir, util.MODEL_DIR_NAME, "best_model.pth")))
model.eval()

count = 0

loss = 0
idx = 0
mse = torch.nn.MSELoss()
for data, target in test_loader:
    idx += 1
    imgs, sps = data.to(DEVICE).detach(), target.to(DEVICE).detach()
    b_size = imgs.size(0)
    g_sps, _, _ = model(sps)
    g_sps = g_sps.detach()
    test_loss = mse(g_sps, sps)
    loss += test_loss
    TEST_LOSS_FORM = "Test Loss: {:.8f}, "
    test_loss_log = TEST_LOSS_FORM.format(test_loss)
    print(test_loss_log)
    with open(os.path.join(log_dir, util.TEST_DIR_NAME, "test_loss.log"), "a") as f:
        f.write(test_loss_log + "\n")
    sps = sps.cpu().numpy()
    g_sps = g_sps.cpu().numpy()
    for i in range(b_size):
        count += 1
        sp = sps[i]
        g_sp = g_sps[i]

        os.makedirs(os.path.join(log_dir, util.TEST_DIR_NAME, str(count)))

        np.savetxt(os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "sp_bc.txt"), sp)
        np.savetxt(os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "g_sp_bc.txt"), g_sp)

        plt.plot(np.linspace(1, 5, int(y_len)), sp, label="SP")
        plt.plot(np.linspace(1, 5, int(y_len)), g_sp, label="G_SP")
        plt.ylim(-scale / 2, scale / 2)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Absorptivity(Box-Cox)')
        plt.legend()
        plt.savefig(os.path.join("./", log_dir, util.TEST_DIR_NAME, str(count) + "_sp_bc.png"))
        plt.close()

        sp = dloader.rev_resolution(np.expand_dims(sp, axis=0))[0]
        g_sp = dloader.rev_resolution(np.expand_dims(g_sp, axis=0))[0]

        np.savetxt(os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "sp.txt"), sp)
        np.savetxt(os.path.join(log_dir, util.TEST_DIR_NAME, str(count), "g_sp.txt"), g_sp)

        plt.plot(np.linspace(1, 5, int(y_len)), sp, label="Simulation")
        plt.plot(np.linspace(1, 5, int(y_len)), g_sp, label="Predict")
        plt.ylim(0, 1)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Absorptivity')
        plt.legend()
        plt.savefig(os.path.join(log_dir, util.TEST_DIR_NAME, str(count) + "_sp.png"))
        plt.close()
print(str(loss / idx))
