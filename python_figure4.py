import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  
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
variables = ['th', 'u', 'v', 'w']#, 'Q']
base_path = f'{base_path}'

qs = [20,15,25,12,28,10,30]  #Q_s = qs / 100
case_indices = [0, 1, 2, 3,4,5,6]
casen = 2*len(qs)
cases = list(range(0, casen)) 
print(casen,cases)


#The data can be found at https://doi.org/10.5281/zenodo.18297902.

input_file_path = f'{base_path}/data_cbl_fLES_q7.npy'  
data = np.load(input_file_path)
print(f"Loaded data_in shape: {data.shape}") 
data_fles = data#[case_indices, ...]    
print(f"data_fles shape: {data_fles.shape}") 

data0 = data_fles[:, 0, :, :, :, :] 
mean_data = np.mean(data0, axis=(0))  
mean_data = np.mean(mean_data, axis=(0))
mean_data = np.mean(mean_data, axis=(0))
mean_data[..., 4] = 0
data0 = mean_data[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  
print("Mean data shape:", mean_data.shape)  
  
input_file_path = f'{base_path}/data_cbl_fno_q7.npy'
data = np.load(input_file_path)
print(f"Loaded data_in shape: {data.shape}") 
data_fno = data[:, :73]        
data_fno = data_fno  + data0
print(f"Loaded data_fno shape: {data_fno.shape}")  
data = np.concatenate((data_fles, data_fno), axis=0)  
print("Combined data shape:", data.shape) 


labels = ['(a)', '(b)', '(c)', '(d)']
y_labels = [r"$E_A (\langle \theta \rangle ) $", r"z (km)", r"$ E_A (\sqrt{\langle w^{\prime} w^{\prime} \rangle })$", r"z (km)" ]
x_labels = [r"t (h)", r"$ \langle \theta \rangle $ (K)", r"t (h)", r"$\sqrt{\langle w^{\prime} w^{\prime} \rangle }$ (m/s )" ]

############################ 
time_steps_to_plot = np.arange(0, 73, 2).tolist()       
num_time_steps = len(time_steps_to_plot)    
mean_ct = np.zeros((len(cases), len(time_steps_to_plot), 32, len(labels)))
plot_val = np.zeros((len(qs), len(time_steps_to_plot), len(labels)))   
for i, time_step in enumerate(time_steps_to_plot):
    for j, case in enumerate(cases):        
        adjusted_case_index = j
        selected_data = data[adjusted_case_index, time_step, :, :, :, 0]   
        mean_val = np.mean(selected_data, axis=(0, 1)).squeeze()
        mean_ct[j,i,:,0] = mean_val
                         
        selected_data = data[adjusted_case_index, time_step, :, :, :,3]                 
        mean_val = np.mean(selected_data, axis=(0, 1)).squeeze() 
        w_fluctuation = selected_data - mean_val[np.newaxis, np.newaxis, :]   
        rms_val = np.sqrt(np.mean(w_fluctuation**2, axis=(0, 1))).squeeze()
        mean_ct[j,i,:,2] = rms_val                                      
print("mean_ct:", mean_ct.shape)
val_fles = mean_ct[0:casen//2, ...]
val_fno = mean_ct[casen//2:casen, ...]
print("val_fles:", val_fles.shape)
dval = np.sqrt((val_fles - val_fno)**2)
mean_z = np.mean(dval, axis=(2)).squeeze()
print("mean_z:", mean_z.shape) 
plot_val = mean_z
print("plot_val:", plot_val.shape)
        
          
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 16))  
z_dim = data.shape[4]  
z_size_km = z_size  
z = np.linspace(0.04, z_size_km, z_dim)  
time_x = np.array(time_steps_to_plot) * 5 / 60   
t_plot = 24
for variable_index, variable in enumerate(labels):
    ax = axs[variable_index // 2, variable_index % 2]
    for j , q in enumerate(qs):
        if j < casen // 2:
            color_index = j 
        else:
            color_index = j - (casen // 2)  
        color = colors[color_index]  
        if j == 0:
            linestyle = line_styles[0]  
        elif j in [1, 2]:
            linestyle = line_styles[1]  
        elif j in [3, 4,5,6]:
            linestyle = line_styles[2]  
            
        if variable_index == 0 :        
            axs[variable_index // 2, variable_index % 2].plot(time_x,plot_val[j, :, variable_index],
                label=f'$Q_s$= 0. {q}',linestyle=linestyle, linewidth=linewidth, color=color)
        elif variable_index == 2: 
            axs[variable_index // 2, variable_index % 2].plot(time_x,plot_val[j, :, variable_index],label=f'$Q_s$= 0. {q}',linestyle=linestyle, linewidth=linewidth, color=color)                   
        elif variable_index == 1:   
            axs[variable_index // 2, variable_index % 2].plot( mean_ct[j,t_plot,:,0],z[:],
                label='fLES' if j == 0 else "",linestyle=linestyle, linewidth=linewidth, color=color)
            axs[variable_index // 2, variable_index % 2].plot( mean_ct[j+casen//2 ,t_plot,:,0],z[:],
                label='FNO' if j == 0 else "",linestyle='None', marker='o', markersize=9, linewidth=linewidth, color=color)
        elif variable_index == 3:  
            axs[variable_index // 2, variable_index % 2].plot( mean_ct[j,t_plot,:,2],z[:],
                label=f'$Q_s$= 0. {q}',linestyle=linestyle, linewidth=linewidth, color=color)
            axs[variable_index // 2, variable_index % 2].plot( mean_ct[j+casen//2 ,t_plot,:,2],z[:],
            linestyle='None', marker='o', markersize=9, linewidth=linewidth, color=color)            
    x_limits = [
            [0, 6],    
            [300, 310],   
            [0, 6],    
            [0, 1.]   
        ]
    y_limits = [
            [0, 0.1],     
            [0, 2.],   
            [0, 0.08],   
            [0, 2]   
        ]
    ax.set_xlabel(x_labels[variable_index],fontsize=40)
    ax.set_xlim(x_limits[variable_index])  
    ax.set_ylim(y_limits[variable_index])
    ax.set_ylabel(y_labels[variable_index],fontsize=40)
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='x', direction='in')  
    ax.grid(False)
    ax.text(-0.3, 1.05, labels[variable_index], transform=ax.transAxes,
                            fontsize=44, va='top')
    
axs[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))
axs[0, 0].legend(loc='upper left', fontsize=28)   
axs[0, 1].legend(loc='upper left', fontsize=28)            

plt.tight_layout()
filename_mean = f"figure4.{file_extension}"
if file_extension == "eps":
    plt.savefig(filename_mean)  
else:
    plt.savefig(filename_mean, dpi=dpi_value)  
plt.show()  
    

    

