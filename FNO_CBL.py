# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os

torch.manual_seed(123)
np.random.seed(123)
################################################################
# 4d fourier layers
class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv4d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul4d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3,], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, nvar):
        super(FNO4d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(nt_in*nvar+3, self.width)  # nt_in time * nvar + 3 grid

        self.conv0 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv4 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        # self.conv5 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)      
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)	
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        # self.w4 = nn.Conv1d(self.width, self.width, 1)
        # self.w5 = nn.Conv1d(self.width, self.width, 1)
        
        self.bn0 = torch.nn.BatchNorm3d(self.width)     
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, nvar)                  

    def forward(self, x):  # [batchsize,Nx,Ny,Nz,nvar,nt_in]
        batchsize = x.shape[0]
        size_x, size_y, size_z= x.shape[1], x.shape[2], x.shape[3]
        # print(x.shape)

        grid = self.get_grid(batchsize, size_x, size_y, size_z, x.device)  
        x = x.permute(0, 1, 2, 3, 5, 4).reshape(batchsize,size_x,size_y,size_z,nvar*nt_in) 
        x = torch.cat((x, grid), dim=-1) 
        x = self.fc0(x) 
        x = x.permute(0, 4, 1, 2, 3) 

        x1 = self.conv0(x) 
        x2 = self.w0(x)  
        # x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z) #conv1d
        x = x1 + x2
        # x = self.bn0(x1 + x2) # normalize
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.bn1(x1 + x2) # normalize
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.bn2(x1 + x2) # normalize
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.bn3(x1 + x2) # normalize
        # x = F.relu(x)
        
        # x1 = self.conv4(x)
        # x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        # x = x1 + x2
        # x = F.relu(x)
        
        # x1 = self.conv5(x)
        # x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        # x = x1 + x2
        
        
        x = x.permute(0, 2, 3, 4, 1) #[bs,32,32,32,width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x  

    def get_grid(self, batchsize, size_x, size_y, size_z, device ):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x,1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])

        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
################################################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
#device = torch.device("mps") 
#-------------------------------------------------------------------------------
var = [0, 1, 2, 3, 4]
# 获取列表的长度
nvar = len(var)
case_start = 0
case_delta = 1
case_all =70
case_train = case_all
time_dstep = 1 
timenumber_be = 0
timenumber = 73
nt_in = 5    # 5 for FNOI5O1, 1 for FNOI1O1
modes = 16   #16, 14,12,10
modes_z = 6 
width = 96   
epochs = 30
epochs_out = 19  # 
learning_rate = 0.001  #0.01-0.0001
weight_decay_value = 1e-8  #-8 or -11
batch_size = 4   
scheduler_step = 8    
scheduler_gamma = 0.5  

filter_type = "sharp"  #   sharp  gaussian  tophat   downsample
file_path = f'/.../cbl_q7n70_Qne2_{filter_type}_z32.npy'

print("Loading data from:", file_path)
print(f" filter_type: {filter_type}, nvar: {nvar}, nt_in: {nt_in}")
print(f" case_all: {case_all}, case_train: {case_train}, case_start: {case_start}, case_delta: {case_delta}")
print(f" time_dstep: {time_dstep}, timenumber_be: {timenumber_be}, timenumber: {timenumber}")
print(f" modes: {modes}, modes_z: {modes_z}, width: {width} ")
print(f" epochs: {epochs}, epochs_out: {epochs_out}, scheduler_step: {scheduler_step}, scheduler_gamma: {scheduler_gamma}")
print(f" learning_rate: {learning_rate}, weight_decay_value: {weight_decay_value}, batch_size: {batch_size} ")
#------------------------------------------------------------------------------
runtime = np.zeros(2, )    
t1 = default_timer()      
################################################################
# load data
################################################################
#-------------------------------------------------
vor_data = np.load(file_path)   
print("Original data shape:", vor_data.shape)
#训练使用数据
vor_data = vor_data[case_start:case_all:case_delta, :, :, :, :, :][:, :, :, :, :, var]
print("Processed data shape:", vor_data.shape)


data = vor_data[:, 0, :, :, :, :] 
mean_data = np.mean(data, axis=(0))  #  (1, 73, 5)
mean_data = np.mean(mean_data, axis=(0))
mean_data = np.mean(mean_data, axis=(0))
print(mean_data.shape)  #  (32（z）, 4)
print(mean_data) 
mean_data[..., 4] = 0.
print(mean_data.shape)  #  (32（z）, 4)
print(mean_data) 

data = mean_data[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  
vor_data = vor_data - data  # shape will be (70, 73, 32, 32, 32, 5)
print("vor_data shape:", vor_data.shape)  

data = vor_data
mean_data = np.mean(data, axis=(0, 2, 3))  
rms_data = np.sqrt(np.mean(data**2, axis=(0, 2, 3))).squeeze()
print(mean_data.shape,rms_data.shape)  
np.save('mean_data_new.npy', mean_data)  
np.save('rms_data_new.npy', rms_data) 

#########################################                
vor_data = torch.from_numpy(vor_data) # 
vor_data = vor_data[:,timenumber_be:timenumber:time_dstep,...]
print(vor_data.shape)
##########################################
input_list = []
output_list = []
####################################################
for j in range(vor_data.shape[0]):
    for i in range(vor_data.shape[1]- nt_in):  
        input_list.append(vor_data[j,i:i+ nt_in,...])  # .append 
        #output_6m5 = (vor_data[j,i+5,...]-vor_data[j,i+4,...])
        #output_list.append(output_6m5)                
        output_list.append(vor_data[j,i+nt_in,...])                           
### switch dimension    
input_set = torch.stack(input_list) 
output_set = torch.stack(output_list) 
input_set = input_set.permute(0,2,3,4,5,1) 
#######################
full_set = torch.utils.data.TensorDataset(input_set, output_set)
train_dataset, test_dataset = torch.utils.data.random_split(full_set, [int(0.8*len(full_set)), 
                                                                       len(full_set)-int(0.8*len(full_set))])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO4d(modes, modes, modes_z, width, nvar).to(device) 
print(modes, modes, modes_z, width, nvar)
print(count_params(model)) 
#################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

mse_train = []
mse_test = []


myloss = LpLoss()  
# myloss = torch.nn.MSELoss(reduction='mean')
for ep in range(epochs+1):  
    model.train()
    t1 = default_timer()
    for xx, yy in train_loader:   
        
        xx = xx.to(device)
        yy = yy.to(device)
        im = model(xx).to(device)
        
        train_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    scheduler.step()
    mse_train.append(train_loss.item())   
        

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            im = model(xx).to(device)
            test_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))
        mse_test.append(test_loss.item())

    t2 = default_timer()
    
    if (ep > epochs_out and (20 <= ep <= 61 and ep % 2 == 0 or ep % 5 == 0)):
        torch.save(model.state_dict(), './model/model_4layer_epochs{}.pth'.format(ep))
    
    print(ep, "%.2f" % (t2 - t1), 'train_loss: {:.4f}'.format(train_loss.item()), 
          'test_loss: {:.4f}'.format(test_loss.item()))

MSE_save=np.dstack((mse_train,mse_test)).squeeze()
np.savetxt('./loss_4layer.txt',MSE_save,fmt="%16.7f")
# redefine retive error function