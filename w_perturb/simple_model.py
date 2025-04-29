import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import kornia

import numpy as np
import matplotlib.pyplot as plt

from auto_LiRPA import BoundedModule, BoundedParameter
from auto_LiRPA.perturbations import PerturbationLpNorm

from LGaussianBlur import LGaussianBlur

def build_bounded_sigma(sigma_lower, sigma_upper):
    center = (sigma_upper + sigma_lower) / 2.0
    diff = (sigma_upper - sigma_lower) / 2.0

    ptb = PerturbationLpNorm(norm = np.inf, eps = diff)

    tensor = BoundedParameter(torch.Tensor([center]), ptb, requires_grad=False)

    return center, tensor

test_data = torchvision.datasets.MNIST(
    './data', train=False, download=True,
    transform=torchvision.transforms.ToTensor())

image = test_data.data[0].expand(1, 1, -1, -1)
image = image.to(torch.float32) / 255.0

kernel_size = 11
sigma_center, sigma_tensor = build_bounded_sigma(0.5, 2.0)

model = nn.Sequential(
    LGaussianBlur(image.size(1), kernel_size, sigma_tensor),
)

lirpa_model = BoundedModule(
    model,
    torch.empty_like(image), device=image.device)

fig, ax = plt.subplots(3, 2)

tv_blur = F.conv2d(image,
                   kornia.filters.get_gaussian_kernel2d(kernel_size, (sigma_center, sigma_center)).expand(1, 1, -1, -1),
                   padding=kernel_size // 2,
                   groups=image.size(1))

lm_blur_lb, lm_blur_ub = lirpa_model.compute_bounds(image, method='IBP')

ax[0][0].imshow(torch.squeeze(image).unsqueeze(-1).detach())
ax[0][1].imshow(torch.squeeze(tv_blur).unsqueeze(-1).detach())
ax[1][0].imshow(torch.squeeze(lm_blur_lb).unsqueeze(-1).detach())
ax[1][1].imshow(torch.squeeze(lm_blur_ub).unsqueeze(-1).detach())
ax[2][0].imshow(torch.abs(torch.squeeze(lm_blur_lb - tv_blur)).unsqueeze(-1).detach())
ax[2][1].imshow(torch.abs(torch.squeeze(lm_blur_ub - tv_blur)).unsqueeze(-1).detach())

plt.show()