from typing import Any

import torch
import torch.nn as nn


class SCO_ReLU(torch.autograd.Function):

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x, s, b):
        y = torch.zeros_like(x)

        mask1 = x < -s + b
        mask3 = x > s - b
        mask2 = (~mask1) & (~mask3)

        y[mask1] = b * torch.exp((x[mask1] + s - b) / b) - s
        y[mask2] = x[mask2]
        y[mask3] = -b * torch.exp((-x[mask3] + s - b) / b) + s

        ctx.s = s
        ctx.b = b
        ctx.save_for_backward(x)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        s = ctx.s
        b = ctx.b

        grad_input = torch.zeros_like(grad_output)

        mask1 = x < -s + b
        mask3 = x > s - b
        mask2 = (~mask1) & (~mask3)

        grad_input[mask1] = grad_output[mask1] * torch.exp((x[mask1] + s - b) / b)
        grad_input[mask2] = grad_output[mask2]
        grad_input[mask3] = grad_output[mask3] * torch.exp((-x[mask3] + s - b) / b)

        return grad_input, None, None


class sco_relu(nn.Module):
    def __init__(self, s, b):
        super(sco_relu, self).__init__()
        if s <= 0:
            raise ValueError("s 的值必须大于0")
        if b > s:
            raise ValueError("b 的值必须小于 s")
        if b < 0:
            raise ValueError("b 的值必须大于 0")
        self.s = s
        self.b = b

    def forward(self, input):
        return SCO_ReLU.apply(input, self.s, self.b)
