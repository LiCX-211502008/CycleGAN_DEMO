from include import *
from torch import device, nn, tensor

class ResBlock(nn.Module):
    def __init__(self,in_features) -> None:
        super(ResBlock,self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,in_features,3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,in_features,3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self,x):
        return self.block(x)
    
class Generator(nn.Module):
    def __init__(self,input_shape,nums_of_ResBlock):
        super(Generator,self).__init__()
        channels = input_shape[0]
        out_features = int(64)
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels,out_features,kernel_size = 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace = True)
        ]
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features,out_features,kernel_size = 3,stride = 1,padding = 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace = True),
            ]
            in_features = out_features
        for _ in range(nums_of_ResBlock):
            model += [ResBlock(out_features)]
        for _ in range(2):
            out_features //= 2
            model +=[
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(in_features,out_features,kernel_size = 3,stride = 1,padding = 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace = True),
            ]
            in_features = out_features
        model +=[
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features,channels,kernel_size = 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self,x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self,input_shape) -> None:
        super(Discriminator,self).__init__()
        channels,height,width = input_shape
        self.output_shape = (1,height // 2 ** 4,width // 2 ** 4)
        def discriminator_block(in_filters,out_filters,normalize = True):
            layers = [nn.Conv2d(in_filters,out_filters,kernel_size = 4,stride = 2,padding = 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2,inplace = True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(channels,64,normalize = True),
            *discriminator_block(64,128),
            *discriminator_block(128,256),
            *discriminator_block(256,512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512,1,4,padding = 1)
        )
    def forward(self,x):
        return self.model(x)
