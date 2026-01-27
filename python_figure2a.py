import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 36  
file_extension = "png"
dpi_value = 300 

colors = ['black','b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']  
line_styles = ['-', '--', '-.']  
linewidth = 3

x_size = 25.6  # km
y_size = x_size 
z_size = 2.56  # km
x_min = 0
x_max = 32
t_delat = 5
variables = ['th', 'u', 'v', 'w']#, 'Q']
base_path = f'{base_path}'

qs = [24,24,24,24,24]  #Q_s = qs / 100
casen = 2*len(qs) +1 
cases = list(range(0, casen)) 
print(casen,cases)

#The data can be found at https://doi.org/10.5281/zenodo.18326720 and at https://doi.org/10.5281/zenodo.18297902.
input_file_path = f'{base_path}/data_cbl_fLES_q1_024.npy'  
data = np.load(input_file_path)
print(f"Loaded data_in shape: {data.shape}") 
data_fles = data[...,:4]#[case_indices, ...]    
print(f"data_fles shape: {data_fles.shape}") 

data0 = data_fles[:, 0, :, :, :, :] 
mean_data = np.mean(data0, axis=(0))  
mean_data = np.mean(mean_data, axis=(0))
mean_data = np.mean(mean_data, axis=(0))
#mean_data[..., 4] = 0
data0 = mean_data[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  
print("Mean data shape:", mean_data.shape)  
  
input_file_path = f'{base_path}/data_cbl_fno_q1_024.npy'
data = np.load(input_file_path)
print(f"Loaded data_in shape: {data.shape}") 
data_fno = data[:, :73,...,:4]        
data_fno = data_fno  + data0
print(f"Loaded data_fno shape: {data_fno.shape}")  

input_file_path = f'{base_path}/data_grayz32.npy'   
data = np.load(input_file_path)
data_gray  = np.expand_dims(data, axis=0)  
print(f"Loaded gray data_in shape: {data_gray.shape}") 

data = np.concatenate((data_fles, data_fno,data_gray), axis=0)  
print("Combined data shape:", data.shape) 

  
time_steps_to_plot = [0, 24, 48, 72] 
for variable_index in range(0, 1):
    variable = variables[variable_index]
    print(variable_index, variable)       
    num_time_steps = len(time_steps_to_plot)   

    plt.figure(figsize=(10, 9))
    z_dim = data.shape[4]  
    z_size_km = z_size
    z = np.linspace(0.04, z_size_km, z_dim)  
    plot_val = np.zeros(( len(cases), len(time_steps_to_plot),  32)) 
    for i, time_step in enumerate(time_steps_to_plot):
        for j, case in enumerate(cases):
            adjusted_case_index = case                 
            selected_data = data[adjusted_case_index, time_step, :, :, :, variable_index]                
            mean_val = np.mean(selected_data, axis=(0, 1)).squeeze()    
            plot_val[j,i,:] = mean_val
    print("mean_val shape:", plot_val.shape)       
    plotval_fles = np.mean(plot_val[0:5, :, :], axis=0)
    print("plotval_fles shape:", plotval_fles.shape)            
    plotval_fno = np.mean(plot_val[5:10, :, :], axis=0)
    print("plotval_fno shape:", plotval_fno.shape)   
    plotval_gray = plot_val[10, :, :]             
    plotval = np.stack((plotval_fles, plotval_fno, plotval_gray ), axis=0)  
    print("plotval shape:", plotval.shape)
    
    for i, time_step in enumerate(time_steps_to_plot):
        plt.plot(plotval[0, i, :], z, linestyle='-', color=colors[i], label=f'fLES' if i == 0 else "")  
        plt.plot(plotval[1, i, :], z, linestyle='None', marker='o',markersize=8,  color=colors[i], label=f'FNO ' if i == 0 else "") 
        plt.plot(plotval[2, i, :], z, linestyle='--', color=colors[i], label=f'CM1 ' if i == 0 else "")  
    fixed_x_values = {
        0: 300, 
        1: 302.2, 
        2: 304.2, 
        3: 306  
    }      
    y_pos = z[6]               
    for i, time_step in enumerate(time_steps_to_plot):
        fixed_x = fixed_x_values.get(i, 0)               
        plt.text(fixed_x+0.2, y_pos, 
                 f'{int(time_step * t_delat / 60)}h', 
                 color=colors[i], 
                 ha='left', 
                 va='bottom')
    plt.text(-0.24, 1.02, '(a)', fontsize=46, color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.ylabel('z (km)')
    plt.xlabel(r'$\langle \theta \rangle $ (K)') 
    plt.ylim(0, 2)  
    plt.xlim(299, 310)  
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))  
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')  
    plt.legend() 
    plt.grid(True)
    plt.tight_layout()
    filename_mean = f"figure2a.{file_extension}"
    if file_extension == "eps":
        plt.savefig(filename_mean)  
    else:
        plt.savefig(filename_mean, dpi=dpi_value)  
    plt.show()  
    
    
