'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F
import model.Conv2d_quadratic as Cq
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet56', 'resnet110']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, Relu=True, quadratic=False, feature_map_size=16):
        super(BasicBlock, self).__init__()
        self.relu = Relu
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.quadratic = quadratic
        if quadratic:
            self.quadratic_layer = Cq.Conv2d_quadratic(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.ln_in = nn.LayerNorm([feature_map_size, feature_map_size])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.id = nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.quadratic:
            out = (self.bn2(self.conv2(out))+self.quadratic_layer(self.ln_in(out)))/2
        else:
            out = self.bn2(self.conv2(out))
        out = self.id(out)
        out += self.shortcut(x)
        if self.relu:
            out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, quadratic=False, feature_map_size=32)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, quadratic=True, feature_map_size=16)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, quadratic=False, feature_map_size=8)
        self.linear = nn.Linear(256*block.expansion, num_classes)
       
    def _make_layer(self, block, planes, num_blocks, stride, Relu=True, quadratic=False, feature_map_size=16):
        strides = [1]*(num_blocks-1)
        layers = []
        layers.append(block(self.in_planes, planes, stride, Relu, quadratic, feature_map_size))
        self.in_planes = planes * block.expansion
        for stride1 in strides:
            layers.append(block(self.in_planes, planes, stride1, Relu))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.contiguous()
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def dit_resnet20():
    return ResNet(BasicBlock, [3,3,3])

def dit_resnet32():
    return ResNet(BasicBlock, [5,5,5])

def dit_resnet56():
    return ResNet(BasicBlock, [9,9,9])

def dit_resnet110():
    return ResNet(BasicBlock, [18,18,18])