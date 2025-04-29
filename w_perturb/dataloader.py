from torchvision import transforms, datasets
import torch
import os

def normalize():
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def unnormalize():
    return transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

def forward_transform(input_size):
    return transforms.Compose([
        transforms.RandomRotation(25),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize()
    ])

def reverse_transform():
    return transforms.Compose([
        unnormalize(),
        transforms.ToPILImage()
    ])

def get_dataloaders(input_size, batch_size, shuffle=True, data_dir='separated-data'):
    '''
    Create the dataloaders for train, validation and test set. Randomly rotate images for data augumentation
    Normalization based on std and mean.

    Parameters:
    - input_size: Size to resize images to
    - batch_size: Batch size for training
    - shuffle: Whether to shuffle the data
    - data_dir: Directory containing the dataset (with train, val, test subdirectories)

    Returns:
    - dataloaders: Dictionary with train, val, test dataloaders
    - class_names: List of class names
    '''

    transform = forward_transform(input_size)

    # Create image datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform)
        for x in ['train', 'val', 'test']
    }

    # Create dataloaders
    num_workers = 8
    print(f"Using {num_workers} workers for DataLoader")

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=shuffle if x == 'train' else False,
            num_workers=num_workers
        )
        for x in ['train', 'val', 'test']
    }

    return dataloaders, image_datasets['train'].classes