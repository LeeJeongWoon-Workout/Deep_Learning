import numpy as np
import torch

#1.PyTorch tensor grammer- very similar to numpy

nums=torch.arange(9)
#[0,1,2,3,4,5,6,7,8]
#torch.arange(n)  (1,n) [0~n-1] matrix
nums.shape
#torch.Size([9])
nums.reshape(3,3)
#(3,3) matrix transform
random=torch.rand((3,3))
#(3,3) matrix random component
zeros=torch.zeros((3,3))
#(3,3)matrix all components are zero
ones=torch.ones((3,3))
#(3,3)matrix all components are ones
zeros2=torch.zeros_like(ones)
#the size of matrix is ones and all components are zeros



#2.Operation
nums=torch.arange(9)
nums+3
#all components + 3
nums=nums.reshape((3,3))
nums + nums
'''tensor([[ 0,  2,  4],
        [ 6,  8, 10],
        [12, 14, 16]])'''

torch.add(nums, 3)
'''tensor([[ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])'''

result = torch.add(nums, 10)
result.numpy()
'''array([[10, 11, 12],
       [13, 14, 15],
       [16, 17, 18]])'''



#3.View
range_nums = torch.arange(9).reshape(3,3)
'''tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])'''

range_nums.view(-1) #rashape과 동일한 기능
'''tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
'''

range_nums.view(3,3)
'''tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])'''

#4.Slice
nums=torch.arange(9).reshape((3,3))
nums[1]
'''tensor([3, 4, 5])
'''

nums[1,1]
'''tensor(4)
'''

nums[1:,1:]
'''tensor([[4, 5],
        [7, 8]])'''

nums[1:]
'''tensor([[3, 4, 5],
        [6, 7, 8]])'''

