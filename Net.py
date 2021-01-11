import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid


class Net(nn.Module):
  def __init__(self, n_ch):
    super(Net, self).__init__()

    self.n_ch = n_ch
    self.conv1 = nn.Conv2d(3, self.n_ch, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(self.n_ch, self.n_ch*2, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(self.n_ch*2, self.n_ch*4, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(self.n_ch*4*8*8, self.n_ch*4)
    self.fc2 = nn.Linear(self.n_ch*4, 6)

  def forward(self, x):
    # img = 3ch x 64 px x 64 px
    x = self.conv1(x) # n_channels x 64 x 64
    x = F.relu(F.max_pool2d(x,2)) # divide img by 2 -> n_channels x 32 x 32
    x = self.conv2(x) # n_channels x 32 x 32
    x = F.relu(F.max_pool2d(x,2)) # divide img by 2 -> n_channels*2 x 16 x 16 
    x = self.conv3(x) # n_channels x 16 x 16
    x = F.relu(F.max_pool2d(x,2)) # divide img by 2 -> n_channels*4 x 8 x 8

    # flatten
    x = x.view(-1, self.n_ch*4*8*8)

    # fc
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)

    # Log_softmax
    x = F.log_softmax(x, dim = 1)

    return x