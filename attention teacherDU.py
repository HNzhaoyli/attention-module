import torchvision.models as models
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import init
from torch.nn import functional as F

#参考论文CBAM_ Convolutional Block Attention Module
#coded by H. Du at henu
#2020.4.2

class channel_attention(nn.Module):
    def __init__(self,in_channel,reduce_ratio):
        super(channel_attention, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.GMP = nn.AdaptiveMaxPool2d((1,1))
        self.sigmoid = nn.Sigmoid()
        self.MLP= nn.Sequential(nn.Linear(in_channel, in_channel//reduce_ratio),
                                nn.LeakyReLU(0.1,inplace=True),nn.Linear( in_channel//reduce_ratio, in_channel))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.constant_(m.bias.data, 0.0)
    def forward(self,x):
        Avg_rst = self.GAP(x)
        Max_rst = self.GAP(x)
        rst = self.sigmoid(Avg_rst + Max_rst)
        rst1 = rst.view(rst.size(0), -1)
        Mc = self.MLP(rst1)
        Mc = Mc.unsqueeze(2)
        Mc = Mc.unsqueeze(3)
        c_out = x * (Mc.expand_as(x))
        return c_out

class spatial_attention(nn.Module):
    def __init__(self,ker_size=7,strid=1,pad=3):
        super(spatial_attention, self).__init__()
        self.Conv2D = nn.Conv2d(2,1,ker_size,strid,pad)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.constant_(m.bias.data, 0.0)
    def forward(self, x):
        Avg_rst = x.mean(dim=1)
        Max_rst, index = x.max(dim=1)
        Avg_rst = Avg_rst.unsqueeze(1)
        Max_rst = Max_rst.unsqueeze(1)

        rst = torch.cat((Avg_rst, Max_rst), 1)

        Ms = self.Conv2D(rst)
        Ms = self.sigmoid(Ms)
        s_out = x * (Ms)
        return s_out

class cs_attention(nn.Module):
    def __init__(self,in_channel,reduce_ratio,ker_size=7,strid=1,pad=3):
        super(cs_attention, self).__init__()
        self.ch_atten = channel_attention(in_channel,reduce_ratio)
        self.sp_atten = spatial_attention(ker_size,strid,pad)

    def forward(self,x):
        x_c = self.ch_atten(x)
        out = self.sp_atten(x_c)
        return out


if __name__ == '__main__':
    inputs = Variable(torch.randn(16,4,32,32))
    cs = cs_attention(4,2)
    out = cs(inputs)
    print(out.shape)









