"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import os
from collections import defaultdict
from auto_LiRPA.bounded_tensor import BoundedParameter
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten

from LGaussianBlur import LGaussianBlur

## Step 1: Define computational graph by implementing forward()
# This simple model comes from https://github.com/locuslab/convex_adversarial
def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

model = mnist_model()
# Optionally, load the pretrained weights.
# checkpoint = torch.load(
#     os.path.join(os.path.dirname(__file__), 'pretrained/mnist_a_adv.pth'),
#     map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint)

## Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST(
    './data', train=False, download=True,
    transform=torchvision.transforms.ToTensor())
# For illustration we only use 2 image from dataset
N = 2
n_classes = 10
image = test_data.data[:N].view(N,1,28,28)
true_label = test_data.targets[:N]
# Convert to float
image = image.to(torch.float32) / 255.0
if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()

def build_bounded_sigma(sigma_lower, sigma_upper):
    center = (sigma_upper + sigma_lower) / 2.0
    diff = (sigma_upper - sigma_lower) / 2.0

    ptb = PerturbationLpNorm(norm = torch.inf, eps = diff)

    tensor = BoundedParameter(torch.Tensor([center]), ptb, requires_grad=False)

    return center, tensor

kernel_size = 11
sigma_center, sigma_tensor = build_bounded_sigma(0.5, 4.0)

augment_model = nn.Sequential(
    LGaussianBlur(image.size(1), kernel_size, sigma_tensor),
    model,
)

## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph,
# and its content is not important.
lirpa_model = BoundedModule(augment_model, torch.empty_like(image), device=image.device)
print('Running on', image.device)
# Visualize the lirpa_model
# Visualization file is saved as "bounded_mnist_model.png" or "bounded_mnist_model.dot"
lirpa_model.visualize("bounded_mnist_model")
print()

## Step 4: Compute bounds using LiRPA given a perturbation
# Get model prediction as usual
pred = lirpa_model(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()
print('Demonstration 1: Bound computation and comparisons of different methods.\n')

## Step 5: Compute bounds for final output
for method in ['IBP']:
    print('Bounding method:', method)
    if 'Optimized' in method:
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
    for i in range(N):
        print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
        for j in range(n_classes):
            indicator = '(ground-truth)' if j == true_label[i] else ''
            print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
    print()

## An example for computing margin bounds.
# In compute_bounds() function you can pass in a specification matrix C, which is a final linear matrix applied to the last layer NN output.
# For example, if you are interested in the margin between the groundtruth class and another class, you can use C to specify the margin.
# This generally yields tighter bounds.
# Here we compute the margin between groundtruth class and groundtruth class + 1.
# If you have more than 1 specifications per batch element, you can expand the second dimension of C (it is 1 here for demonstration).
lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
C = torch.zeros(size=(N, 1, n_classes), device=image.device)
groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
target_label = (groundtruth + 1) % n_classes
C.scatter_(dim=2, index=groundtruth, value=1.0)
C.scatter_(dim=2, index=target_label, value=-1.0)
print('Demonstration 3: Computing bounds with a specification matrix.\n')
print('Specification matrix:\n', C)

for method in ['IBP']:
    print('Bounding method:', method)
    if 'Optimized' in method:
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
    for i in range(N):
        print('Image {} top-1 prediction {} ground-truth {}'.format(i, label[i], true_label[i]))
        print('margin bounds: {l:8.3f} <= f_{j}(x_0+delta) - f_{target}(x_0+delta) <= {u:8.3f}'.format(
            j=true_label[i], target=(true_label[i] + 1) % n_classes, l=lb[i][0].item(), u=ub[i][0].item()))
    print()
