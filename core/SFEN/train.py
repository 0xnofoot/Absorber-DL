import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from core.dt.absorber.loader_build import Dataloader_BoxCox, Dataset_BoxCox
from model import sfen, loss_func
import util

DEVICE = util.DEVICE


def train(model, tloader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(tloader):
        sp = target.to(DEVICE)
        optimizer.zero_grad()
        g_sp, mean, logvar = model(sp)
        loss = criterion(g_sp, sp, mean, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(tloader)

    return train_loss


def epoch_train(tloader, epoch_count, lr, log_dir_name=None):
    model, latent_dim = sfen.get_sfen()
    model = model.to(DEVICE)

    log_dir = util.create_log_dir(log_dir_name)

    criterion = loss_func.vae_lf()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_loss = float("inf")

    for epoch in range(1, epoch_count + 1):
        train_loss = train(model, tloader, optimizer, criterion)
        best_loss = util.log_handle(epoch, train_loss, best_loss, model,
                                    log_dir, scheduler_lr.get_last_lr())
        scheduler_lr.step()
    util.save_loss_pic(log_dir)


def train_this(epoc=500, lr=0.001, b_size=64):
    train_loader, val_loader, _ = Dataloader_BoxCox("cross_data").get()
    dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    d_loader = DataLoader(dataset, batch_size=b_size, shuffle=True, drop_last=True, pin_memory=True)
    epoch_train(tloader=d_loader, epoch_count=epoc, lr=lr)


if __name__ == "__main__":
    train_this()
