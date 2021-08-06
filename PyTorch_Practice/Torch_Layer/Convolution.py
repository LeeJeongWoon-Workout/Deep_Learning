import torch
import torch.nn as nn
import torch.nn.functional as F

nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
'''in_channels: 받게 될 channel의 갯수
out_channels: 보내고 싶은 channel의 갯수
kernel_size: 만들고 싶은 kernel(weights)의 사이즈'''

layer = nn.Conv2d(1, 20, 5, 1).to(torch.device('cpu'))
layer

weight = layer.weight
weight.shape
type(weight)
weight = weight.detach().numpy()
'''여기서 weight는 학습 가능한 상태이기 때문에 바로 numpy로 뽑아낼 수 없음
detach() method는 그래프에서 잠깐 빼서 gradient에 영향을 받지 않게 함'''

weight.shape
'''(20, 1, 5, 5)
'''

plt.imshow(weight[0,0,:,:],'jet')
plt.colorbar()
plt.show()

output_data = layer(image)
output = output_data.data.cpu().numpy()
output.shape
'''(1, 20, 24, 24)
'''

plt.figure(figsize=(15,30))
plt.subplot(131)
plt.title('Input')
plt.imshow(np.squeeze(image_arr), 'gray') #pyplot 에 들어가기 위해서 squeeze 해야함 (2,2)
plt.subplot(132)
plt.title('Weight')
plt.imshow(weight[0,0,:,:], 'jet')
plt.subplot(133)
plt.title('Output')
plt.imshow(output[0,0,:,:], 'gray')
plt.show()
