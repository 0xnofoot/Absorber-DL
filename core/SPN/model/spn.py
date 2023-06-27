import torch
import torch.nn as nn
from . import ggn, ac


# SPN 正常大小
def get_spn(num_classes=1000):
    return SPN_0(num_classes=num_classes), num_classes


class SPN_0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.main = nn.Sequential(
            ggn.GoogleNet_353(num_classes=num_classes),
            ac.sco_relu(s=1, b=0.09),
        )

    def forward(self, img):
        img = nn.functional.interpolate(img, size=(256, 256))
        img = torch.cat([img, img, img], dim=1)
        g_sp = self.main(img)
        return g_sp
