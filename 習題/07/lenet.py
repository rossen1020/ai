import torch.nn as nn
import torch.nn.functional as F
import sys

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)   #將kernal_size修改至就可以得到97%的正確率
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)  #將kernal_size修改至就可以得到97%的正確率
        self.fc1 = nn.Linear(160, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print('x.shape=', x.shape)
        # sys.exit(1)
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
