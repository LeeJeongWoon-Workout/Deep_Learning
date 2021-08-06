import torch

from torchvision import datasets, transforms

batch_size = 32
test_batch_size = 32

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/content/dataset', train=True, download=True,
                   transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(mean=(0.5,), std=(0.5,))      
                   ])),
                   batch_size=batch_size,
                   shuffle=True)


test_loader=torch.utils.data.DataLoader(
    datasets.MNIST('/content/dataset',train=False,
                   transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,),(0.5))
                   ])),
    batch_size=test_batch_size,
    shuffle=True
)

images, labels = next(iter(train_loader))
images.shape
'''torch.Size([32, 1, 28, 28])
    [Batch Size, Channel, Height, Width] '''
'''There are 32 images only 1 color in images '''
labels.shape
images[0].shape
'''torch.Size([1, 28, 28])
'''
#Data Visualization
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

torch_image=torch.squeeze(images[0])
torch_image.shape

#pyplot only sense (2,2) matrix thus eliminate Channel
image=torch_image.numpy()
image.shape

label = labels[0].numpy()

plt.title(label)
plt.imshow(image, 'gray')
plt.show()
