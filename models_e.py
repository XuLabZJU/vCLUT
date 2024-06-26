import torch.nn as nn
import torch
import numpy as np
import trilinear

def discriminator_block(in_filters, out_filters, normalization=False):             
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class Classifier_selfpaired(nn.Module):    
    def __init__(self, in_channels=3):   
        super(Classifier_selfpaired, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'), 
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),         
            nn.Conv2d(128, 3, 8, padding=0),   
        )                                       

    def forward(self, img_input):
        return self.model(img_input)

class Generator3DLUT_identity(nn.Module):  
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("./utils/BasisIdentityLUT.txt", 'r')
        lines = file.readlines()     
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)
        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split() 
                    buffer[0,i,j,k] = float(x[0])
                    buffer[1,i,j,k] = float(x[1])
                    buffer[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))   
        self.TrilinearInterpolation = TrilinearInterpolation()  

    def forward(self, x):  
        _, output = self.TrilinearInterpolation(self.LUT, x)     
        return output

class Generator3DLUT_zero(nn.Module):      
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()
        self.LUT = torch.zeros(3,dim,dim,dim, dtype=torch.float) 
        self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):    
        _, output = self.TrilinearInterpolation(self.LUT, x)    
        return output

class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()
        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        assert 1 == trilinear.forward(lut, 
                                      x, 
                                      output,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        ctx.save_for_backward(*variables)
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        assert 1 == trilinear.backward(x, 
                                       x_grad, 
                                       lut_grad,
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad

class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)
