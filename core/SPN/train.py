import torch
import torch.nn as nn
import torch.optim as optim

from core.dt.absorber.loader_build import Dataloader_BoxCox, Dataset_BoxCox
from model import spn
import util

DEVICE = util.DEVICE


def train(model, tloader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(tloader):
        img, sp = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        g_sp = model(img)
        loss = criterion(g_sp, sp)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(tloader)

    return train_loss


def val(model, vloader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in vloader:
            img, sp = data.to(DEVICE), target.to(DEVICE)
            g_sp = model(img)
            loss = criterion(g_sp, sp).item()
            val_loss += loss

    val_loss /= len(vloader)
    return val_loss


def epoch_train(tloader, vloader, epoch_count, lr, log_dir_name=None):
    model, num_classes = spn.get_spn()
    model = model.to(DEVICE)

    log_dir = util.create_log_dir(log_dir_name)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_loss = float("inf")

    for epoch in range(1, epoch_count + 1):
        train_loss = train(model, tloader, optimizer, criterion)
        val_loss = val(model, vloader, criterion)
        best_loss = util.log_handle(epoch, train_loss, val_loss, best_loss, model, log_dir, scheduler_lr.get_last_lr())
        scheduler_lr.step()

    util.save_loss_pic(log_dir)


def train_this(epoc=1000, lr=0.001):
    train_loader, val_loader, _ = Dataloader_BoxCox("cross_data").get()
    epoch_train(tloader=train_loader, vloader=val_loader, epoch_count=epoc, lr=lr)


if __name__ == "__main__":
    train_this()
