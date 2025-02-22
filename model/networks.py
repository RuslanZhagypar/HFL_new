import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNEMnist(nn.Module):
    def __init__(self, args):
        #TODO: number of channels is fixed in this model now
        super(CNNEMnist, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(7 * 7 * 32, 256)
        self.fc2 = nn.Linear(256, 62)

    def forward(self, x):
        conv1_ = self.conv1(x)
        conv2_ = self.conv2(conv1_)
        fc_ = conv2_.view(-1, 32*7*7)
        fc1_ = self.fc1(fc_).clamp(min=0)  # Achieve relu using clamp
        output = self.fc2(fc1_)
        return output
