import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 36  # 可根据需要调整字体大小
file_extension = "png"
dpi_value = 300  # 指定 DPI


colors = ['k','r','b', 'g',  'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']  # 可以根據需要添加更多顏色
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

th = data[..., 0]  
u = data[..., 1]   
v = data[..., 2]   
w = data[..., 3]  
                      
data = w

labels = ['(a-1)', '(a-2)', '(a-3)']
titles = ["w', 120 m", "w', 600 m", "w', 1240 m"]

z_layers = [1,7,15]
time_steps_to_plot = [24, 60]
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 24))  # 竖着排列的子图

for z_layer_index, z_layer in enumerate(z_layers):
    plot_val = np.zeros((len(cases), len(time_steps_to_plot), 22))
    rms_val = np.zeros(len(cases))
    for i, time_step in enumerate(time_steps_to_plot):
        for j, case in enumerate(cases):
            adjusted_case_index = case
            f_slice = data[adjusted_case_index, time_step, :, :, z_layer]
            mean_val = np.mean(f_slice, axis=(0, 1)).squeeze()
           
            fluctuation = f_slice - mean_val[np.newaxis, np.newaxis]
            rms_val[j] = np.mean(fluctuation**2, axis=(0, 1)).squeeze()         
        ke_fles = np.mean(rms_val[0:5], axis=0)
        print(time_step, ke_fles)
        
        for j, case in enumerate(cases):
            adjusted_case_index = case      
            selected_data = data[adjusted_case_index, time_step, :, :, z_layer]
            mean_val = np.mean(selected_data, axis=(0, 1)).squeeze() 
            if j == 10:
                print(mean_val)
            f_slice  = selected_data  - mean_val[np.newaxis, np.newaxis]
            fft_data = np.fft.fft2(f_slice)
            power_spectrum = np.abs(fft_data) ** 2
            power_spectrum_shifted = np.fft.fftshift(power_spectrum)

            Y, X = np.indices(power_spectrum_shifted.shape)
            radius = np.sqrt((X - X.mean())**2 + (Y - Y.mean())**2).astype(int)
            wave_number = radius / np.sqrt(2)
            radial_profile = np.bincount(radius.ravel(), power_spectrum_shifted.ravel()) / np.bincount(radius.ravel())
            plot_val[j, i, :] = radial_profile / 32/32 #/  ke_les[z_layer_index,i]

    plotval_fles = np.mean(plot_val[0:5, :, :], axis=0)
    plotval_fno = np.mean(plot_val[5:10, :, :], axis=0)
    plotval_gray = plot_val[10, :, :]  
    plotval = np.stack((plotval_fles, plotval_fno, plotval_gray), axis=0)
    
    for i, time_step in enumerate(time_steps_to_plot):
        x_values = np.arange(1, plotval.shape[2]+1 )  
        #xles_values = np.arange(1, plotval_les.shape[2]+1 )
    
        if z_layer_index == 0:
            print(i, z_layer_index)
            #axs[z_layer_index].plot(xles_values, plotval_les[z_layer_index, i, :], linestyle='-', color=colors[i], label='LES' if i == 0 else "")
            axs[z_layer_index].plot(x_values, plotval[0, i, :], linestyle='--', 
                     linewidth=4, color=colors[i], label='fLES' if i == 0 else "")
            axs[z_layer_index].plot(x_values, plotval[1, i, :], linestyle='None', marker='o', markersize=9, color=colors[i], label='FNO' if i == 0 else "")
            axs[z_layer_index].plot(x_values, plotval[2, i, :], linestyle='None', marker='o', markersize=9, markerfacecolor='none', color=colors[i], label='CM1' if i == 0 else "")
            axs[z_layer_index].text(3., 0.000003, r'$\Delta_h$ = 4 km', fontsize=30, verticalalignment='center')
            axs[z_layer_index].text(11., 0.00001, r'$\Delta_h$ = 800 m', fontsize=30, verticalalignment='center')
        else:
            #axs[z_layer_index].plot(xles_values, plotval_les[z_layer_index, i, :], linestyle='-', color=colors[i], label='LES' if i == 0 else "")
            axs[z_layer_index].plot(x_values, plotval[0, i, :], linestyle='--',linewidth=4, color=colors[i])
            axs[z_layer_index].plot(x_values, plotval[1, i, :], linestyle='None', marker='o', markersize=9, color=colors[i])
            axs[z_layer_index].plot(x_values, plotval[2, i, :], linestyle='None', marker='o', markersize=9, markerfacecolor='none', color=colors[i])
            axs[z_layer_index].text(3., 0.000003, r'$\Delta_h$ = 4 km', fontsize=30, verticalalignment='center')
            axs[z_layer_index].text(11., 0.00001, r'$\Delta_h$ = 800 m', fontsize=30, verticalalignment='center')
        for x in [4.5,  22]:  
            axs[z_layer_index].axvline(x=x, linestyle='dotted', color='gray', linewidth=1)
        
    if z_layer_index == 0:    
        fixed_x_values = {
            0: 10,  
            1: 10,  
            2: 0.3,  
        }   
        fixed_y_values = {
            0: 2.3,  
            1: 0.6, 
            2: 0.3, 
        }   
        for i, time_step in enumerate(time_steps_to_plot):
            fixed_x = fixed_x_values.get(i, 0)    
            fixed_y = fixed_y_values.get(i, 0) 
            axs[z_layer_index].text(fixed_x + 3., fixed_y - 0.1, f'{int(time_step * t_delat / 60)}h', color=colors[i], ha='left',  va='bottom')
        
    axs[z_layer_index].set_xlim(0.89, 250)
    axs[z_layer_index].set_ylim(1e-6, 1e1)
    axs[z_layer_index].set_xscale('log')
    axs[z_layer_index].set_yscale('log')
    axs[z_layer_index].set_xlabel(r'$ k_h $', labelpad=-10)
    axs[z_layer_index].set_ylabel(r'$ E_w $', labelpad=0)
    if z_layer_index == 0:
        axs[z_layer_index].legend(loc='lower center', bbox_to_anchor=(0.805, 0.42)) 
    axs[z_layer_index].grid(False)    
    axs[z_layer_index].text(-0.2, 1.12, labels[z_layer_index], transform=axs[z_layer_index].transAxes,fontsize=44, va='top')
    axs[z_layer_index].set_title(titles[z_layer_index], loc='center', pad=20)
    axs[z_layer_index].tick_params(axis='both', direction='in', length=6) 
    axs[z_layer_index].tick_params(axis='both', which='minor', direction='in', length=4)  

plt.tight_layout()
filename_mean = f"figure3a.{file_extension}"
if file_extension == "eps":
    plt.savefig(filename_mean) 
else:
    plt.savefig(filename_mean, dpi=dpi_value)  
plt.show()  
       
