from torch.nn import ConvTranspose2d as up
import torch
import torch.nn as nn
#两次3X3的卷积操作
class Doubleconv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channel), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channel), 
            nn.ReLU(inplace=True)
        )
    def forward(self,input):
        return self.conv(input)
class UNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1):
        super().__init__()
        #下采样
        self.d1=Doubleconv(in_channels,64)
        self.p1=nn.MaxPool2d(2)
        self.d2=Doubleconv(64,128)
        self.p2=nn.MaxPool2d(2)
        self.d3=Doubleconv(128,256)
        self.p3=nn.MaxPool2d(2)
        self.d4=Doubleconv(256,512)
        self.p4=nn.MaxPool2d(2)
        #最下面中间连接
        self.d5=Doubleconv(512,1024)
        #上采样
        self.u1=up(1024,512,2,stride=2,padding=0)
        self.c1=Doubleconv(1024,512)
        self.u2=up(512,256,2,stride=2,padding=0)
        self.c2=Doubleconv(512,256)
        self.u3=up(256,128,2,stride=2,padding=0)
        self.c3=Doubleconv(256,128)
        self.u4=up(128,64,2,stride=2,padding=0)
        self.c4=Doubleconv(128,64)
        #输出结果
        self.out=nn.Conv2d(64,out_channels,1)
    #裁剪拼接，跳跃连接
    def cp(self,target,input):
        target_height,target_width=target.size(2),target.size(3)
        input_height,input_width=input.size(2),input.size(3)
        crop_height=(input_height-target_height)//2
        crop_width=(input_width-target_width)//2
        return input[:,:,crop_height:crop_height+target_height,crop_width:crop_width+target_width]
    def forward(self,x):
        d1=self.d1(x)
        d2=self.d2(self.p1(d1))
        d3=self.d3(self.p2(d2))
        d4=self.d4(self.p3(d3))
        d5=self.d5(self.p4(d4))
        u1=self.u1(d5)
        c_d4=self.cp(u1,d4)
        out1=self.c1(torch.cat([c_d4,u1],dim=1))
        u2=self.u2(out1)
        c_d3=self.cp(u2,d3)
        out2=self.c2(torch.cat([c_d3,u2],dim=1))
        u3=self.u3(out2)
        c_d2=self.cp(u3,d2)
        out3=self.c3(torch.cat([c_d2,u3],dim=1))
        u4=self.u4(out3)
        c_d1=self.cp(u4,d1)
        out4=self.c4(torch.cat([c_d1,u4],dim=1))
        out=self.out(out4)
        return out