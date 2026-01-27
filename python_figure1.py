import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 36  
file_extension = "png"
dpi_value = 100  

colors = ['k','r','b', 'g',  'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']  
colors = ["blue", "white", "red"]
custom_cmap = LinearSegmentedColormap.from_list("custom_red_blue", colors)
cmap= custom_cmap  

x_size = 25.6  
y_size = x_size 
z_size = 2.56  
x_min = 0
x_max = 25.6


base_path = f'{base_path}'

#The data can be found at https://doi.org/10.5281/zenodo.18326720 and at https://doi.org/10.5281/zenodo.18297902.
input_file_path = f'{base_path}/cm1cbl_les_predata.npy'  
data = np.load(input_file_path)
print(f"Loaded data_in shape: {data.shape}") 
data_les = data[:, :, :, 1::4, :]
print(f" data_les shape: {data_les.shape}")  

input_file_path = f'{base_path}/data_cbl_fLES_q1_024.npy'   
data = np.load(input_file_path)
print(f"Loaded data_in shape: {data.shape}") 
data_fles = data[:,::12,..., :4]
print(f"data_fles shape: {data_fles.shape}")  

data0 = data_fles[:, 0, :, :, :, :] 
mean_data = np.mean(data0, axis=(0))  
mean_data = np.mean(mean_data, axis=(0))
mean_data = np.mean(mean_data, axis=(0))
data0 = mean_data[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  
print("Mean data shape:", mean_data.shape)  

input_file_path = f'{base_path}/data_cbl_fno_q1_024.npy'
data = np.load(input_file_path)
print(f"Loaded data_in shape: {data.shape}") 
data_fno = data[:, ::12, ..., :4] 
data_fno = data_fno  + data0
print(f"data_fno shape: {data_fno.shape}")  

input_file_path = f'{base_path}/data_grayz32.npy'   
data = np.load(input_file_path)
data = data[::12,...]
data_gray  = np.expand_dims(data, axis=0) 
print(f"gray data shape: {data_gray.shape}") 

data = np.concatenate((data_fles, data_fno, data_gray), axis=0)  
print("Combined data shape:", data.shape)  
        
              
var  = 3 # w   
z_layers = [1]
time_steps_to_plot = [4]
case_plots = [11,2 , 7, 10]
case_plots = [11,1 , 6, 10]
case_plots = [11,3 , 8, 10]
case_plots = [11,4 , 9, 10]
case_plots = [11,0 , 6, 10]
titles = ['(a) LES', ' (b) fLES', '(c) FNO', '(d) CM1']

for z_layer in z_layers:
    for time_step in time_steps_to_plot:
        print(f"Current z_layer: {z_layer} {time_step}")
        ncols = len(case_plots)
        nrows = 1  
        fig, axs = plt.subplots(nrows, ncols , figsize=(32, 8))  
        axs = axs.flatten()    

        for i, case_plot in enumerate(case_plots):
            print(f"case_plot: {case_plot}  {i} {time_step}")
            if i == 0:
                f_slice = data_les[time_step, :, :, z_layer,var]
                x_dim = data_les.shape[1]
                y_dim = data_les.shape[2]
                z_dim = data_les.shape[3]
                x = np.linspace(0, x_size, x_dim)
                y = np.linspace(0, y_size, y_dim)
                z = np.linspace(0, z_size, z_dim)
                print(f"x shape: {x.shape}, y shape: {y.shape}, z shape: {z.shape}") 
            else:
                f_slice = data[case_plot,time_step, :, :, z_layer, var] 
                x_dim = data.shape[2]
                y_dim = data.shape[3]
                z_dim = data.shape[4]
                x = np.linspace(0, x_size, x_dim)
                y = np.linspace(0, y_size, y_dim)
                z = np.linspace(0, z_size, z_dim)
                print(f"x shape: {x.shape}, y shape: {y.shape}, z shape: {z.shape}")  
            print(f"f_slice shape: {f_slice.shape}")
            all_max = np.max(f_slice[:, :])
            all_min = np.min(f_slice[:, :])
            if all_max == all_min:
                all_max = all_min + all_min / 10
                print(f"if Max = {all_max}, Min = {all_min}")
            max_abs_index = np.unravel_index(np.argmax(np.abs(f_slice)), f_slice.shape)
            max_abs_value = f_slice[max_abs_index]
            all_max = max_abs_value if max_abs_value > 0 else -max_abs_value
            all_min = -all_max
            levels = np.linspace(all_min, all_max , 300)
       
            contour = axs[i].contourf(x, y, f_slice, levels=levels, cmap=cmap,  alpha=1) #cmap='jet',
            title_obj = axs[i].set_title(titles[i], fontsize=54, pad=30)
            title_obj.set_position((-0.13, 1.))  
            axs[i].set_xlabel('x', fontdict={'fontstyle': 'italic'}, fontsize=50, labelpad=-5)
            axs[i].set_ylabel('y', fontdict={'fontstyle': 'italic'}, fontsize=50, labelpad=-0)
            axs[i].set_xlim(x_min , x_max )           
            axs[i].set_xticks(np.linspace(x_min, x_max , num=5))  
            axs[i].set_yticks(np.linspace(x_min, x_max , num=5))  
            axs[i].tick_params(axis='both', which='major')   
            axs[i].text(0.00, 1.00, f'Max: {np.max(f_slice):.2f}\nMin: {np.min(f_slice):.2f}', 
                        transform=axs[i].transAxes, verticalalignment='top', 
                        bbox=dict(facecolor='white', alpha=0.64, edgecolor='none'))       
            
            divider = make_axes_locatable(axs[i])

            cax = divider.append_axes("top", size="5%", pad=0.6)
            cbar = plt.colorbar(contour, cax=cax, orientation='horizontal')
            cbar.ax.set_aspect(0.1)  
            cbar.ax.tick_params(direction='in')    
            contour.set_clim(all_min, all_max)
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}" if x == 0 else f"{x:.2f}".rstrip("0").rstrip(".")))
           
            if i == 0:
                cbar.set_ticks([-3.0, 0, 3.0])  
            elif i == 1:
                cbar.set_ticks([-1, 0, 1])
            elif i == 2:
                cbar.set_ticks([-1, 0, 1])
            elif i == 3:
                cbar.set_ticks([-1., 0, 1.])
       
        filename = f'figure1.{file_extension}'
        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.13)
        plt.savefig(filename, dpi=dpi_value)  
        plt.show()           
 



           
            
