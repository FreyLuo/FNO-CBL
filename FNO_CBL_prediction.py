import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch
#from utilities3 import *

torch.manual_seed(123)
np.random.seed(123)

# os.chdir(r'')
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
    
device = torch.device("cuda")
#**************************************************************************
#-----------------------------------
var = [0, 1, 2, 3, 4]
nvar = len(var)
time_dstep = 1 
timenumber_bes = [0]
timenumber = 73
modes = 16  #16, 14, 12, 10 
modes_z = 6 
width = 96
Nx = 32
Ny = 32
Nz = 32

nt_in = 5  # 5 for FNOI5O1, 1 for FNOI1O1
##### For Q_s = 0.24, 
case_indices = [0, 1, 2, 3,4]
##### For Q_s = 0.20, 0.15, 0.25, 0.12, 0.28
case_indices = [5, 6, 7, 8,9]

casen = len(case_indices)
file_path = f'/.../data_cbl_fLES.npy'


print("Loading data from:", file_path)
print(f"  nvar: {nvar}, nt_in: {nt_in}")
print(f" time_dstep: {time_dstep}, timenumber_be: {timenumber_bes}, timenumber: {timenumber}")
print(f" modes: {modes}, modes_z: {modes_z}, width: {width} ")
print(f" Nx: {Nx}, Ny: {Ny}, Nz: {Nz}")

################################################################
# load data
vor_data_all = np.load(file_path)   
print("Original data shape:", vor_data_all .shape)    # [5, 73, 32, 32, 32, 5] 
#### data.shape[0] = casen: different surface heat fluxes Q_s
#### data.shape[1] = 73 : 73 time slices.  
#### data.shape[2],data.shape[3],data.shape[4] = 32 : NX, NY, NZ
#### data.shape[5] = 5 : potential temperature $\theta$, three velocity components $u_i$, exponential heat fluxes $Q$

for t_be, timenumber_be in enumerate(timenumber_bes): 
    step_adv = timenumber - nt_in  - timenumber_be
    vor_data = vor_data_all[case_indices, 0:timenumber:time_dstep, :, :, :, :][:, :, :, :, :, var]
    print("Processed data shape:", vor_data.shape)   

    data = vor_data[:, 0, :, :, :, :] 
    mean_data = np.mean(data, axis=(0)) 
    mean_data = np.mean(mean_data, axis=(0))
    mean_data = np.mean(mean_data, axis=(0))
    print(mean_data.shape)  
    print(mean_data) 
    mean_data[..., 4] = 0.
    print(mean_data.shape)  
    print(mean_data) 
    np.save('mean_data.npy', mean_data) 
    data = mean_data[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  
    vor_data_1to1 = vor_data - data  
    print("Mean data shape:", mean_data.shape)  
    print("vor_data shape:", vor_data.shape)  

    data = vor_data_1to1
    # #------------------------------------------------------------------------------------------------
    vor_data_1to1 = torch.from_numpy(vor_data_1to1) 
    vor_data  =  vor_data_1to1[:,timenumber_be:timenumber_be+nt_in,...]              
    print(f"vor_data_1to1: {vor_data_1to1.shape}") 
    print(f"vor_data: {vor_data.shape}")
    #######################################
    model = FNO4d(modes, modes, modes_z, width, nvar).to(device)       
    for step_out in range(30, 31): 
        #if step_out > 19 and (24 <= step_out <= 60 and step_out % 2 == 0 or step_out % 5 == 0):
            print(step_out)
            PATH = '../model/model_4layer_epochs{}.pth'.format(step_out)
            model.load_state_dict(torch.load(PATH))  
            model.eval() 
            ###########################################################################
            pre_vor_t_total = torch.zeros([1,Nx,Ny,Nz,nvar,step_adv])  #initial 
            sample_id_data = list(range(casen))
            time_id_data =[0]
            for sample_id in sample_id_data:
                for time_id in time_id_data:   
                    time_advanced_step = step_adv  
                    input_vor = vor_data[sample_id,time_id:time_id+nt_in,...].permute(1,2,3,4,0) 
                    label_vor = vor_data[sample_id,time_id+nt_in:time_id+nt_in+time_advanced_step,...].permute(1,2,3,4,0) 
                    ####################
                    pre_vor_t = input_vor[:,:,:,:,0].unsqueeze(-1)  
                    for i in range(time_advanced_step):  
                        predict_vor = model(input_vor.unsqueeze(0).to(device)).squeeze().detach().cpu()    
                        if nt_in == 5:  
                            pre_vor_t = torch.cat((pre_vor_t, predict_vor.unsqueeze(-1)),dim=-1)  
                            #print(f"pre_vor_t: {pre_vor_t.shape}")
                            #print(f" {pre_vor_t[0,0,0,:,:]}")  
                            vor_new = torch.cat((input_vor[:,:,:,:,1:],predict_vor.unsqueeze(-1)),dim=-1) 
                            input_vor = vor_new 
                            input_vor[...,4,4] = input_vor[...,4,0]
                        elif nt_in == 1:  
                            #print(f"pre_vor_t: {predict_vor.shape}")
                            #print(f" {predict_vor[0,0,0:32:8,:]}")  
                            vor_new = torch.cat((input_vor[:,:,:,:,1:],predict_vor.unsqueeze(-1)),dim=-1) 
                            input_vor = vor_new
                        #print(i)
                        #print(f"input_vor: {input_vor.shape}")
                        #print(f" {input_vor[0,0,0:32:8,4,:]}") 
                        
                    pre_vor_t = pre_vor_t[:,:,:,:,1:] 
                    pre_vor_t_total = torch.cat((pre_vor_t_total, pre_vor_t.unsqueeze(0)),dim=0)
                    
            pre_vor_t_total = pre_vor_t_total[1:,...]
            print(f"pre_vor_t_total: {pre_vor_t_total.shape}")
            
            start_vor = vor_data_1to1[:, 0:timenumber_be+nt_in, ...].permute(0, 2, 3, 4,5, 1)  
            print(f"start_vor: {start_vor.shape}")

            new_pre_vor_t_total =  torch.zeros([casen,Nx,Ny,Nz,nvar,step_adv+timenumber_be+nt_in])  

            new_pre_vor_t_total[..., :timenumber_be+nt_in] = start_vor  
            
            new_pre_vor_t_total[..., timenumber_be+nt_in:] = pre_vor_t_total   

            pre_vor_t_total = new_pre_vor_t_total

            print(f"pre_vor_t_total: {pre_vor_t_total.shape}")
            
            #-----------------------------------------------------------------------------------------------------continue times output
            #######################################
            pre_vor_t_total = pre_vor_t_total.permute(0,5,1,2,3,4) #
            print(f"pre_vor_t_total: {pre_vor_t_total.shape}")
            numpy_pre_vor_t_total = pre_vor_t_total.detach().cpu().numpy() 
            print(f"numpy_pre_vor_t_total: {numpy_pre_vor_t_total.shape}")

            file_path = 'predicted_data_epochs{}.npy'.format(timenumber_be, step_out)
            
            if not os.path.exists(file_path):
                directory = os.path.dirname(file_path)
                os.makedirs(directory, exist_ok=True)
                
                np.save(file_path, numpy_pre_vor_t_total)
                print(f"File created at: {file_path}")
            else:
                print(f"File already exists at: {file_path}")
                np.save(file_path, numpy_pre_vor_t_total)
            
            print(f"Data saved to {file_path}")