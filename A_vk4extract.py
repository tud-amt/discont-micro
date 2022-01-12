# Python VK4 raw image extraction

import numpy as np
from PIL import Image
import vk4extract #<----module has to be in the same folder as this file to run
import os

# Change directory to the folder with Keyence .vk files os.chdir(r'C:\Users\fill\in\path\to\folder\with\all\VK4-files') root = ('.\\')
vkimages = os.listdir(root) # Lists all the files in the folder as VKimages

#loops through vkimages to find .vk4 files and extract RGB data
for img in vkimages: 
    if (img.endswith('.vk4')): 
        with open(img, 'rb') as in_file:
            offsets = vk4extract.extract_offsets(in_file) 
            rgb_dict = vk4extract.extract_color_data(offsets, 'peak', in_file) # use extract_img_data for light and height data
            
            rgb_data = rgb_dict['data'] 
            height = rgb_dict['height'] 
            width = rgb_dict['width']
            rgb_matrix = np.reshape(rgb_data, (height, width, 3)) 
            image = Image.fromarray(rgb_matrix, 'RGB') 
            # extract raw RGB images
            image.save(img.replace('.vk4', '.png'), 'PNG') 
            # saves the images as PNG in same folder as root