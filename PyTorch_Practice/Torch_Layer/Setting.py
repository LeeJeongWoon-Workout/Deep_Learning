#Setting
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/content/dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=1)

image, label = next(iter(train_loader))
image.shape, label.shape
plt.imshow(image[0,0,:,:], 'gray')
plt.show()
