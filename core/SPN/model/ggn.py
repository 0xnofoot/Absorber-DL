import torch.nn as nn
import torch.nn.functional as F
from . import block


# A B C 三个 stage 中，每个 stage 的 inception 块的数量为 3 5 3
# 大模型的实现(其实也不大，3 5 3 就是 GoogLeNet v1 的设计）
class GoogleNet_353(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize a GoogleNet_353 instance.
        
        This constructor builds the GoogLeNet-inspired architecture with the following components:
          - Two initial convolutional layers with batch normalization (conv1 and conv2).
          - Three inception blocks (inception3_a, inception3_b, inception3_c) composing stage 3.
          - Three inception blocks (inception4_a, inception4_b, inception4_c) composing stage 4, with updated parameters in inception4_b and inception4_c.
          - Three inception blocks (inception5_a, inception5_b, inception5_c) composing stage 5.
          - A fully connected layer (fc) that maps the flattened final feature vector to the specified number of classes.
        
        Parameters:
          num_classes (int): The number of target classes for the final prediction.
        """
        super().__init__()
        self.conv1 = block.BN_Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.conv2 = block.BN_Conv2d(64, 192, 3, stride=1, padding=1, bias=False)
        self.inception3_a = block.Inception(192, 16, 32, 32, 64, 32, 64)
        self.inception3_b = block.Inception(192, 16, 32, 96, 128, 32, 64)
        self.inception3_c = block.Inception(256, 32, 96, 128, 192, 64, 128)
        self.inception4_a = block.Inception(480, 16, 48, 96, 208, 64, 192)
        self.inception4_b = block.Inception(512, 32, 64, 144, 288, 64, 112)
        self.inception4_c = block.Inception(528, 32, 128, 160, 320, 128, 256)
        self.inception5_a = block.Inception(832, 32, 128, 160, 320, 128, 256)
        self.inception5_b = block.Inception(832, 32, 128, 160, 320, 128, 256)
        self.inception5_c = block.Inception(832, 48, 128, 192, 384, 128, 384)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        Computes the forward pass for the GoogleNet_353 model.
        
        This method defines the sequence of operations for propagating an input tensor through the network.
        The process includes initial convolutional layers with max pooling, followed by three sequential stages of
        inception blocks interleaved with pooling operations. Specifically, the forward method:
          1. Applies a convolution (conv1) and subsequent max pooling.
          2. Processes the result with a second convolution (conv2) and max pooling.
          3. Passes the data through three inception blocks in stage 3 (inception3_a, inception3_b, inception3_c),
             and applies max pooling.
          4. Processes the data through three inception blocks in stage 4 (inception4_a, inception4_b, inception4_c),
             followed by max pooling.
          5. Passes the output through three inception blocks in stage 5 (inception5_a, inception5_b, inception5_c).
          6. Reduces the spatial dimensions with average pooling, applies dropout with a probability of 0.4 (active during training),
             flattens the tensor, and finally maps the features to the class scores via a fully connected layer.
        
        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
        
        Returns:
            torch.Tensor: Output tensor containing class scores for each input sample.
        """
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
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception5_a(out)
        out = self.inception5_b(out)
        out = self.inception5_c(out)
        out = F.avg_pool2d(out, out.size(3))
        out = F.dropout(out, 0.4, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
