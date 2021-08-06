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


#Layer
import torch
import torch.nn as nn
import torch.nn.functional as F

nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
'''in_channels: input channel
out_channels: output channel
kernel_size: kernel(weights)'''

layer = nn.Conv2d(1, 20, 5, 1).to(torch.device('gpu'))
layer

weight = layer.weight
weight.shape
'''torch.Size([20, 1, 5, 5])
  [output_channel,input_channel,kernel_size()]
'''

