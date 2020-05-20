import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        sigmoid_out = self.sigmoid(out)
        out_c_a = x*sigmoid_out
        return out_c_a

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7,padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)#16*1*32*32
        max_out, _ = torch.max(x, dim=1, keepdim=True)#16*1*32*32
        x1 = torch.cat([avg_out, max_out], dim=1)#16*2*32*32
        x2 = self.conv1(x1)#16*1*32*32
        x3 = self.sigmoid(x2)
        out_s_a = x*x3
        return out_s_a


class cs_attention(nn.Module):
    def __init__(self,inplaces,radio=2,kerner_size=7,stride=1,padding=3):
        super(cs_attention,self).__init__()
        self.channelattention = ChannelAttention(inplaces)
        self.spatialattention = SpatialAttention()

    def forward(self, x):
        outc = self.channelattention(x)
        outs = self.spatialattention(outc)
        return  outs

if __name__ == '__main__':
    input = torch.randn(16,4,32,32)
    example  = cs_attention(4)
    out = example(input)
    print(out.size())
