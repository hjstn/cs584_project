import torch
import torch.nn as nn
import torch.nn.functional as F


class LGaussianBlur(nn.Module):
    def __init__(self, kernel_size: int, sigma: torch.Tensor):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.sigma = sigma

        self.sigma_2d = sigma.expand(2).unsqueeze(0)

    def get_gaussian_kernel1d(self):
        device = self.sigma.device
        dtype = self.sigma.dtype if torch.is_floating_point(self.sigma) else torch.float32

        mean = float(self.kernel_size // 2)
        x = (torch.arange(self.kernel_size, device=device, dtype=dtype) - mean)

        gauss = torch.exp(-x.pow(2.0) / (2 * self.sigma.pow(2.0)))

        return gauss / gauss.sum(-1, keepdim=True)
    
    def get_gaussian_kernel2d(self):
        kernel_1d = self.get_gaussian_kernel1d()

        return kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(-2)

    def forward(self, x, *args):
        channels = x.size(1)

        kernel_2d = self.get_gaussian_kernel2d()
        kernel = kernel_2d.expand((1, 1, -1, -1))

        return F.conv2d(x, kernel, padding=self.padding, groups=channels)