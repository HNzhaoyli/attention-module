import torch
import torch.nn  as  nn

input = torch.randn(16,4,32,32)

# channel  attention
MLP = nn.Sequential(nn.Linear(4,2),nn.LeakyReLU(0.1,inplace=True),nn.Linear(2,4))
sigmoid = nn.Sigmoid()
avg_pool = nn.AdaptiveAvgPool2d((1,1))
max_pool = nn.AdaptiveMaxPool2d((1,1))
out_avg = MLP(avg_pool(input).squeeze())
out_max = MLP(max_pool(input).squeeze())
out = out_avg+out_max
finall_rst = sigmoid(out)
print(finall_rst.size())

#spartial attention
cov3 = nn.Conv2d(2,1,7,1,3)
avg_out = torch.mean(input,dim=1,keepdim=True)#16*1*32*32
max_out,_ = torch.max(input,dim=1,keepdim=True)
com = torch.cat((avg_out,max_out),dim=1)#[16, 2, 32, 32]
out=cov3(com)#[16, 1, 32, 32]
final = sigmoid(out)
finall_rst = finall_rst.unsqueeze(2)
finall_rst = finall_rst.unsqueeze(3)
c_out = input*finall_rst
s_out = c_out*final
print(c_out.size(),s_out.size())



