from torch import nn

def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleConv, self).__init__()
        self._conv1 = conv3x3(in_channels, 8, 1)
        self._conv2 = conv3x3(8, out_channels, 1)
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._relu(self._conv1(x))
        x = self._relu(self._conv2(x))
        return x
        

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_batch_norm=False):
        self._conv1 = conv3x3(in_channels, mid_channels, stride=1)
        self._conv2 = conv3x3(mid_channels, out_channels, stride=1)
        self._use_batch_norm = use_batch_norm
        self._relu = nn.ReLU()

        if (self._use_batch_norm):
            self._bn1 = nn.BatchNorm2d(in_channels)
            self._bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        short_cut = x
        out = self._conv1(x)
        if (self._use_batch_norm):
            out = self._bn1(out)
        out = self._relu(out)

        out = self._conv2(x)
        if (self._use_batch_norm):
            out = self._bn2(out)
        out += short_cut
        out = self._relu(out)
        return out

class ResNet(nn.Module):
    def __init_(self, in_channels, mid_channels, out_channels, use_batch_norm=False):
        self._block1 = ResBlock(in_channels, mid_channels, mid_channels, use_batch_norm)
        self._block2 = ResBlock(mid_channels, out_channels, out_channels, use_batch_norm)
    
    def forward(self, x):
        x = self._block1(x)
        x = self._block2(x)




