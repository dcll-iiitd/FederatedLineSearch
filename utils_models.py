from utils_libs import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, vocab_size=68, embed_dim=32, hidden_dim=64):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out





# class FEMNIST_CNN(nn.Module):
#     def __init__(self, num_classes=62):
#         super(FEMNIST_CNN, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.fc1 = nn.Linear(64 * 7 * 7, 2048)
#         self.fc2 = nn.Linear(2048, num_classes)

#     def forward(self, x):
#         # Handle input reshaping if input is flattened
#         if x.ndim == 3 and x.shape[-1] == 784:
#             x = x.view(-1, 1, 28, 28)
#         elif x.ndim == 4 and x.shape[1:] == (1, 1, 784):
#             x = x.view(-1, 1, 28, 28)
        
#         x = self.pool1(F.relu(self.conv1(x)))  # → [B, 32, 14, 14]
#         x = self.pool2(F.relu(self.conv2(x)))  # → [B, 64, 7, 7]
#         x = x.view(x.size(0), -1)              # → [B, 64*7*7]
#         x = F.relu(self.fc1(x))                # → [B, 2048]
#         x = self.fc2(x)                        # → [B, num_classes]
#         return x


class FedNet(nn.Module):
    def __init__(self,n_c):
        super(FedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384)
        self.fc2 = nn.Linear(384,192)
        self.fc3 = nn.Linear(192, n_c)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EMNIST_CNN(nn.Module):
    def __init__(self):
        super(EMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(12*12*64, 128)
        self.fc2 = nn.Linear(128, 62)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)


    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = func.relu(self.conv1(x))
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2,planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2,self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def resnet18(n_c):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=n_c)


class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionPyTorch, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(784, output_dim)
    
    def forward(self, x):
        x = self.flatten(x)
        return F.log_softmax(self.linear(x), dim=1)
def get_model(model,n_c):
  if(model=='resnet18'):
    return resnet18(n_c)
  elif(model == 'CNN'):
    return FedNet(n_c)
  elif(model=='EMNIST_CNN'):
    return EMNIST_CNN()
  elif model == 'LOGISTIC_REGRESSION':
      return LogisticRegressionPyTorch(input_dim=784, output_dim=10)
  elif model == 'LSTM':
        return LSTMModel()