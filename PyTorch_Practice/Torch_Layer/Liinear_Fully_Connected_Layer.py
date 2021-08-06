image.shape
'''torch.Size([1, 1, 28, 28])
'''

flatten = image.view(1, 28 * 28 )
flatten.shape
'''torch.Size([1, 784])
'''

lin = nn.Linear(784, 10)(flatten)
lin.shape
'''torch.Size([1, 10])
'''

plt.imshow(lin.detach().numpy(), 'jet') # lin에는 weight가 있기 때문에 detach해줘야 에러 x
plt.show()

with torch.no_grad():
  flatten = image.view(1, 28 * 28)
  lin = nn.Linear(784, 10)(flatten)
  softmax = F.softmax(lin, dim=1)
#결과를 numpy로 꺼내기 위해선 weight가 담긴 Linear에 weight를 꺼줘야함

softmax

