import torch
import torch.nn as nn

# 假设这是你的模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.register_buffer("memory_bank", torch.zeros([3, 3], requires_grad=False))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x

# 实例化模型
model = MyModel()
print(model.state_dict().keys())
# for each in model.conv1.named_buffers():
#     print(each)