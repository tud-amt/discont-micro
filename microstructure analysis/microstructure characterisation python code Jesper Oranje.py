
"""
Copyright (c) 2021 Jesper Oranje and Clemens Dransfeld
MIT License

@author: Jesper Oranje
This code has been writen for the authors master thesis at TUDelft which can be found in the TUDelft repository.

Input file for this code is a .csv file created by FIJI ImageJ particle analysis.
The FIJI ImageJ particle analysis pipeline and macro code is given and further explained in the authors Master thesis.  
The .csv file needs to contain the columns 'X','Y','Major','Minor','Angle'
    'X' & 'Y' are the pixel coordinates for every fibre centre location
    'Major' & 'Minor' are the axis length in pixels for the ellipse major and minor axis
    'Angle' is the angle in radians of the major axis relattive to the horizontal axis in the micrograph
 
Advised is to run this code on a cell by cell basis in spyder.
This can be done by selecting the cell and pressing Ctrl+Enter or by clicking on the "run current cell" icon in the toolbar.

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# =============================================================================
#  reads csv file and creates new column "out-of plane"
# =============================================================================
"""
resolution 20x mag --> 1.452 pixels = 1 micron
resolution 50x original --> 3.63 pixels = 1 micron
Depending on which magnification is used during optical microscopy, the amount of pixel per micron as to be added.
This is because the resolution is used to scale certain plots.

20x magnification micrographs are too low a resolution for carbon fibre thus can't be used for measurment of fibre diameters. 
"""
# input
raw_data = pd.read_csv(r'C:\Users\path\to\FIJI_ImageJ\Result_file.csv', usecols=['X','Y','Major','Minor','Angle'])
sample_name, resolution, color = "C1 x50 - 2 layer vacuum ", 3.63, "tab:green"

raw_data["Out-of plane angle"] = np.degrees(np.arccos(raw_data['Minor'] / raw_data['Major'])) # adds the OoP angle to the dataframe
raw_data["Area"] = np.pi*(0.5*raw_data['Minor'])*(0.5*raw_data['Major']) # adds individuel fibre areas to the dataframe

# %%
# =============================================================================
# histograms Out-of-PLane fibre orientation
# =============================================================================
"""
This cell calculates and shows the Out-of-PLane fibre orientation histograms.
mean, median and standard deviation are also calculated and shown. 
The normal distribution representation is shown in histogram by the shaded bins  
"""
bins = np.linspace(0, 90, 90)
sns.set_style('darkgrid')
ax = sns.distplot(raw_data["Out-of plane angle"], bins=bins)

ax.set_xticks(range(0,95,5))
ax.set_title('Histogram: Out of plane Thèta [degree] \n '+str(sample_name))
ax.set_xlabel("Angle [degrees]")

ax = sns.kdeplot(raw_data["Out-of plane angle"], fill=False, label= str(sample_name))

kdeline = ax.lines[0]
xs = kdeline.get_xdata()
ys = kdeline.get_ydata()
mean = raw_data["Out-of plane angle"].mean()
middle = raw_data["Out-of plane angle"].median()
sdev = raw_data["Out-of plane angle"].std()
left = middle - sdev
right = middle + sdev
ax.set_title('Histogram: Out of plane Thèta [degree] \n '+str(sample_name))
ax.set_xlabel("Angle [degrees]")

#fixes the lgend to the upper right corner with the mean, median and std info
anc = AnchoredText(r"mean = {}, std= {}" "\n" "median = {}µm ".format(round(mean,2),round(sdev,2),round(middle,2)), loc="upper right")
ax.add_artist(anc)
#colours the std part of the histogram in a crimson colour for visual clarification 
ax.vlines(middle, 0, np.interp(middle, xs, ys), color='crimson', ls=':')
ax.fill_between(xs, 0, ys, alpha=0.2)
ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), interpolate=True, facecolor='crimson', alpha=0.5)

# %% 
# =============================================================================
# Fibre angle orientation plot 
# =============================================================================
"""
This cell plots the fibre angle orientation distribution plot.
this plot is a visual tool to help indentify which areas in the tape crosssection have varying orientations
"""
ax = plt.scatter(raw_data['X'], (-1*raw_data['Y']), c=raw_data["Out-of plane angle"])
plt.title("orientation plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

cbar = plt.colorbar()
cbar.set_label("orientation out of plane Thèta [grad]")

# %% 
# =============================================================================
# Fibre point position distribution and histogram of X & Y axis plot 
# =============================================================================
"""
This cell plots the point position distribution of the tape cross-section.
The plot also gives a histogram on the horizontal and vertical axis can which help indentify 
which areas in the tape crosssection have higher fibre density fibre packing.
"""
sns.jointplot(x=raw_data['X'], y=(-1*raw_data['Y']))
plt.show()

# %% 
# =============================================================================
# histograms length minor axis
# =============================================================================
"""
This cell calculates and plots the minor axis length histogram.
The minor axis of a fibre cross-section equals it's diameter.
mean, median and standard deviation are given.
It is important to set the right pixel per micron resolution in the very first input cell.
"""
x_minor_min = 5 # lower limit for diamter in µm (carbon fibre was studied)
x_minor_max = 9 # upper limt 
bins = np.linspace(x_minor_min, x_minor_max, 100)
ax = sns.distplot(raw_data['Minor']/resolution, bins=bins, kde=True)  

ax.set_xticks(range(x_minor_min,x_minor_max,1))
ax.set_title('Histogram: length minor axis \n ' +str(sample_name))
ax.set_xlabel("length [micrometer]")
ax.set_xlim([x_minor_min, x_minor_max])

mean = (raw_data['Minor'].mean())/resolution
median = (raw_data['Minor'].median())/resolution
std = (raw_data['Minor'].std())/resolution
anc = AnchoredText(r"mean = {} µm, std= {}µm " "\n" "median = {}µm ".format(round(mean,2),round(std,2),round(median,2)), loc="upper left")
ax.add_artist(anc)

# %% 
# =============================================================================
# histograms minor and major axis
# =============================================================================
"""
This cell calculates and shows the minor and major axis histogram.
it gives the mean and median values for both the minor and major axi.
50x magnification original pixel resolution resolution xlim = minor 19 - 32  , major = 21 - 34 
"""
x_minor_min = 19        
x_minor_max = 32
mean_minor = raw_data['Minor'].mean()
median_minor = raw_data['Minor'].median()
bins = np.linspace(x_minor_min, x_minor_max, 100)
ax = sns.distplot(raw_data['Minor'], bins=bins,label='minor axis', kde=True)
ax.set_xticks(range(x_minor_min,x_minor_max,1))

x_major_min = 21
x_major_max = 34
mean_major = raw_data['Major'].mean()
median_major = raw_data['Major'].median()
bins2 = np.linspace(x_major_min, x_major_max, 100)

ax = sns.distplot(raw_data['Major'], bins=bins2, label='major axis', kde=True)
ax.set_xticks(range(x_major_min,x_major_max,1))
ax.set_title('Histogram: minor & major axis \n ' +str(sample_name))
ax.set_xlabel("length [Pixels]")
ax.set_xlim([x_major_min, x_major_max])

anc = AnchoredText(r"mean minor = {}, major = {} " "\n" "median minor = {}, major = {} ".format(round(mean_minor,2),round(mean_major,2),round(median_minor,2),round(median_major,2)), loc="upper left")
ax.add_artist(anc)
ax.legend(loc="right")

ax.set_xlim([x_minor_min, x_major_max])

#%% 
# =============================================================================
# Calculate global + local volume fraction + line plot
# =============================================================================
"""
This cell first calcualtes the global volume fraction, it is used as line visualisation.

The other and more important part of this code calculates and plots the local fibre volume fraction behaviour
along the width and through thickness of the sample. 
A 1%  width and thickness bin step size or square size can be choosen.

resolution 20x mag 1.452 pixels = 1 micron
resolution 50x original 3.63 pixels = 1 micron

resolution 20x mag is too low to accuratly measure minor-major axis and thus fibre surface
instead for every x-y coordinate a average fibre diameter (fibre_dia) is used to calculate fibre surface and with this local vf
no in or out-of-plane fibre angle is taken into account with this Simplification.

"""
#   global volume fraction
raw_data["Area"] = np.pi*(0.5*raw_data['Minor'])*(0.5*raw_data['Major'])
tot_area_ellipse = raw_data['Area'].sum()

x_grid_length = raw_data['X'].max()-raw_data['X'].min()
y_grid_length = raw_data['Y'].max()-raw_data['Y'].min()

tot_area = x_grid_length*y_grid_length
global_vol_frac = tot_area_ellipse / tot_area 


    # use when analysing with 20x magnification
# fibre_dia_micron = 6.8 # micron    average dia from 50x measurements
# fibre_dia_pix = fibre_dia_micron * resolution # pixels
# raw_data["Area"] = np.pi*(fibre_dia_pix**2/4)  # surface area of the fibre in pixels

    # step size of bin 1% of complete width or height
x_step_size = raw_data['X'].max()*0.01 
y_step_size = raw_data['Y'].max()*0.01

    # square bin size 
# x_step_size = y_step_size = 40 # [ 400  x 40 pixels] for 20X mag
# x_step_size = y_step_size = 100 # [ 100  x 100 pixels] for origional 50X mag

    # local vf along X width
plt.figure(1)
df_x_area = raw_data.groupby(pd.cut(raw_data['X'], bins=np.arange(raw_data['X'].min(),  
                                                                  round((raw_data['X'].max()), -1),    
                                                                  x_step_size)))['Area'].agg(['sum'])
df_x_area = df_x_area['sum']/(x_step_size*raw_data['Y'].max())
ax = df_x_area.plot(ylabel='Fibre area fraction', grid=True, color=color,   
                    title= 'Local fibre volume fraction along width  \n '+str(sample_name), label= str(sample_name))
plt.axhline(global_vol_frac, color=color) # draw line at global volume fraction
plt.ylim(0, )  
plt.xticks(color='w')
ax.legend()

    # local vf along Y thickness
plt.figure(2)
df_y_area = raw_data.groupby(pd.cut(raw_data['Y'], bins=np.arange(raw_data['Y'].min(),  
                                                                  round((raw_data['Y'].max()), -1),   
                                                                  y_step_size)))['Area'].agg(['sum'])
df_y_area = df_y_area['sum']/(y_step_size*raw_data['X'].max())
ax = df_y_area.plot(ylabel='Fibre area fraction', grid=True,rot=90,color=color,
                    title= 'Local fibre volume fraction through thickness \n '+str(sample_name),   
                    label=str(sample_name))
plt.axhline(global_vol_frac,color=color) # draw line at global volume fraction
plt.ylim(0,)
plt.xticks(color='w')
plt.yticks(rotation=90)
ax.legend()

# %%
# =============================================================================
# Plot local volume fraction
# =============================================================================
"""
This cell calculates and plots the local fibre volume fraction as a grid plot.
This visual tool can help identify microstructural behaviour.

The plotting can be done in a square grid wise fashion or in a 1% width or height respective step size 

resolution 20x mag is too low to accuratly measure minor-major axis thus surface area is useless
instead for every x-y coordinate a average fibre diameter (fibre_dia) is used to calculate fibre surface and with this local vf
"""

# fibre_dia_micron = 6.8 # micron    average dia from 50x measurements
# fibre_dia_pix = fibre_dia_micron * resolution # pixels  (9.8736)
# raw_data["Area"] = np.pi*(fibre_dia_pix**2/4)  # surface area of the fibre in pixels

    # square bin size 
# x_step_size = y_step_size = x_length_grid = y_length_grid = 40 # [ 100  x 100 pixels] for 20X mag
x_step_size = y_step_size = x_length_grid = y_length_grid = 100     # [ 100  x 100 pixels] for origional 50X mag

# step size [pixels] of bin 1% of complete width or height
# x_step_size = raw_data['X'].max()*0.01
# y_step_size = raw_data['Y'].max()*0.01
# x_length = (raw_data['X'].max() - raw_data['X'].min()) 
# y_length = (raw_data['Y'].max() - raw_data['Y'].min())
# x_length_grid = x_length/x_step_size
# y_length_grid = y_length/y_step_size

surface_area_grid =int(x_length_grid*y_length_grid)

plt.hist2d(raw_data['X'], raw_data['Y'], weights = raw_data['Area']/surface_area_grid,   
           bins= [np.arange(0,round((raw_data['X'].max()),-1)+x_step_size, x_step_size),   
                  np.arange(0,round((raw_data['Y'].max()),-1)+y_step_size, y_step_size)])

plt.xlim(min(raw_data['X']),max(raw_data['X']))
plt.ylim(max(raw_data['Y']),min(raw_data['Y']))  # flip y-axis

plt.title("Local fibre volume fraction")
plt.xlabel("width [pixels]")
plt.ylabel("Height [pixels]")

# grabs current axis (GCA) and sets aspect ratio to equal
axes=plt.gca()
axes.set_aspect("equal")

## add colorbar horizontally underneath the x-axis 
cb = plt.colorbar(orientation = 'horizontal')
cb.set_label('Local fibre volume fraction per bin')


# %%  
# =============================================================================
# calculate four nearest neighbours and their distance to one another 
# =============================================================================
"""
This cell calculates and plots the distance of the four nearest neighbours.
The KDTree calculates the distance between point (X&Y coordinates)
The distance between fibres is the distance between fibre surfaces 
because the mean fibre diameter is subtracted from the distance between two points.
this introduces a small biased.
"""
from scipy.spatial import KDTree
data = raw_data.to_numpy()
xy = data[:,:2] # select the X and Y colums
kdtree = KDTree(xy) 

nearest_distance =[]
for i in range(len(xy)):
    d, i = kdtree.query((xy[i]),k=5)  # k=5 for the 4 nearest neighbours because we ignore it self 
    nearest_distance.append(d[1:])  # ignore itself as nearest neighbour

distance_first_nearest_neighbours     = []
distance_second_nearest_neighbours    = []
distance_third_nearest_neighbours     = []
distance_fourth_nearest_neighbours    = []
for i in range(len(nearest_distance)):
    distance_first_nearest_neighbours.append(nearest_distance[i][0])
    distance_second_nearest_neighbours.append(nearest_distance[i][1])
    distance_third_nearest_neighbours.append(nearest_distance[i][2])
    distance_fourth_nearest_neighbours.append(nearest_distance[i][3])
    
    ## chance scale size/pixels when using origional resolution image
distance_first_nearest_neighbours_micrometre = np.array(distance_first_nearest_neighbours)/resolution
distance_second_nearest_neighbours_micrometre = np.array(distance_second_nearest_neighbours)/resolution
distance_third_nearest_neighbours_micrometre = np.array(distance_third_nearest_neighbours)/resolution
distance_fourth_nearest_neighbours_micrometre = np.array(distance_fourth_nearest_neighbours)/resolution

mean = (raw_data['Minor'].mean())/resolution      # mean fibre diameter

# plots the first, second and third nearest neighbours in a CDF histogram
ax = sns.kdeplot(distance_first_nearest_neighbours_micrometre  - mean, cumulative=True, label='First neighbours')
ax = sns.kdeplot(distance_second_nearest_neighbours_micrometre - mean, cumulative=True, label='Second neighbours')
ax = sns.kdeplot(distance_third_nearest_neighbours_micrometre  - mean, cumulative=True, label='Third neighbours')
ax = sns.kdeplot(distance_fourth_nearest_neighbours_micrometre - mean, cumulative=True, label='Fourth neighbours')
ax.set_title('Cumulative density function nearest neigbours \n '+str(sample_name))
ax.set_xlabel("distance [micrometre]")     #  when scale is known
# ax.set_xlabel("distance [pixels]")          #  when scale is unknown

# plt.vlines(x=[(1*mean),(np.sqrt(3)*mean),(2*mean),(np.sqrt(7)*mean)],ymin=0,ymax=1, label='theoretical nearest neighbour distance', linestyles='dotted')

ax.set_xticks(range(0,31,1))
ax.set_xlim([-0.3, 30])

first_percentile80  = np.percentile(distance_first_nearest_neighbours_micrometre, 80) - mean
second_percentile80 = np.percentile(distance_second_nearest_neighbours_micrometre, 80) - mean
third_percentile80  = np.percentile(distance_third_nearest_neighbours_micrometre, 80) - mean
fourth_percentile80 = np.percentile(distance_fourth_nearest_neighbours_micrometre, 80) - mean

anc = AnchoredText(r"Fibre diameter = {} µm " "\n"  
                   "80th percentile neighbour " "\n"   
                   "          First = {} µm" "\n"
                   "          Second = {} µm" "\n"
                   "          Third = {} µm" "\n"
                   "          Fourth = {} µm"
                   .format(round(mean,2),round(first_percentile80,2),round(second_percentile80,2),round(third_percentile80,2),round(fourth_percentile80,2)), loc="right")
ax.add_artist(anc)

ax.legend()

