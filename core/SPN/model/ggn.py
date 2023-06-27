import torch.nn as nn
import torch.nn.functional as F
from . import block


# A B C 三个 stage 中，每个 stage 的 inception 块的数量为 3 5 3
# 大模型的实现(其实也不大，3 5 3 就是 GoogLeNet v1 的设计）
class GoogleNet_353(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = block.BN_Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.conv2 = block.BN_Conv2d(64, 192, 3, stride=1, padding=1, bias=False)
        self.inception3_a = block.Inception(192, 16, 32, 32, 64, 32, 64)
        self.inception3_b = block.Inception(192, 16, 32, 96, 128, 32, 64)
        self.inception3_c = block.Inception(256, 32, 96, 128, 192, 64, 128)
        self.inception4_a = block.Inception(480, 16, 48, 96, 208, 64, 192)
        self.inception4_b = block.Inception(512, 24, 64, 112, 224, 64, 160)
        self.inception4_c = block.Inception(512, 24, 64, 128, 256, 64, 128)
        self.inception4_d = block.Inception(512, 32, 64, 144, 288, 64, 112)
        self.inception4_e = block.Inception(528, 32, 128, 160, 320, 128, 256)
        self.inception5_a = block.Inception(832, 32, 128, 160, 320, 128, 256)
        self.inception5_b = block.Inception(832, 32, 128, 160, 320, 128, 256)
        self.inception5_c = block.Inception(832, 48, 128, 192, 384, 128, 384)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception3_a(out)
        out = self.inception3_b(out)
        out = self.inception3_c(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception4_a(out)
        out = self.inception4_b(out)
        out = self.inception4_c(out)
        out = self.inception4_d(out)
        out = self.inception4_e(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception5_a(out)
        out = self.inception5_b(out)
        out = self.inception5_c(out)
        out = F.avg_pool2d(out, out.size(3))
        out = F.dropout(out, 0.4, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
