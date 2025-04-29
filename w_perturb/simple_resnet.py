import os
import multiprocessing

import tqdm

import torch
import torch.nn as nn
from torchvision import models

from auto_LiRPA import BoundedModule, BoundedParameter
from auto_LiRPA.perturbations import PerturbationLpNorm

from LGaussianBlur import LGaussianBlur
from dataloader import get_dataloaders


def build_bounded_sigma(sigma_lower, sigma_upper):
    center = (sigma_upper + sigma_lower) / 2.0
    diff = (sigma_upper - sigma_lower) / 2.0

    ptb = PerturbationLpNorm(norm = torch.inf, eps = diff)

    tensor = BoundedParameter(torch.Tensor([center]), ptb, requires_grad=False)

    return center, tensor

def build_resnet_model(num_classes):
  resnet = models.resnet50(weights=None)

  num_features = resnet.fc.in_features

  resnet.fc = nn.Sequential(
    nn.Identity(), # replaces dropout
    nn.Linear(num_features, num_classes)
  )

  return resnet, num_features

def perturb_model(model, channels, kernel_size, sigma_tensor):
   return nn.Sequential(
      LGaussianBlur(channels, kernel_size, sigma_tensor),
      model
   ).to(next(model.parameters()).device)

# Set up multiprocessing for Windows
if __name__ == '__main__':
    # Required for Windows when using multiprocessing
    multiprocessing.freeze_support()

    # Use CUDA if available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    data_dir = 'separated-data'

    num_classes = 5
    batch_size = 1
    shuffle_datasets = True

    kernel_size = 3
    sigma_center, sigma_tensor = build_bounded_sigma(0.0, 2.0)

    model, num_features = build_resnet_model(num_classes)
    model.load_state_dict(torch.load('weights/adv1_100', map_location=device))

    dataloaders, class_names = get_dataloaders(num_features, batch_size, shuffle_datasets, data_dir=data_dir)

    sample_inputs, sample_labels = next(iter(dataloaders['train']))

    input_size = sample_inputs[0].unsqueeze(0).size()

    perturbed_model = perturb_model(model, input_size[1], kernel_size, sigma_tensor)
    perturbed_model.eval()

    lirpa_model = BoundedModule(perturbed_model,
                                torch.empty(input_size),
                                device=device)

    lirpa_model.visualize("bounded_perturbed_resnet_model")