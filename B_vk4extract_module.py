# Python module VK4 raw image extraction
""" vk4extract
Original authors
Wylie Gunn & Behzad Torkian
"""
import logging
import struct
import numpy as np
# import readbinary as rb

log = logging.getLogger('vk4_driver.vk4extract')

# extract offsets for data sections of vk4 file
def extract_offsets(in_file):
    """extract_offsets
    Extract offset values from the offset table of a vk4 file. Stores offsets
    and returns values in dictionary
    :param in_file: open file obj, must be vk4 file
    """ 
    log.debug("Entering extract_offsets()")
    offsets = dict()
    in_file.seek(12) 
    offsets['meas_conds'] = struct.unpack('<I', in_file.read(4))[0] 
    offsets['color_peak'] = struct.unpack('<I', in_file.read(4))[0] 
    offsets['color_light'] = struct.unpack('<I', in_file.read(4))[0] 
    offsets['light'] = struct.unpack('<I', in_file.read(4))[0]
in_file.seek(8, 1) 
offsets['height'] = struct.unpack('<I', in_file.read(4))[0]
in_file.seek(8, 1) 
offsets['clr_peak_thumb'] = struct.unpack('<I', in_file.read(4))[0] 
offsets['clr_thumb'] = struct.unpack('<I', in_file.read(4))[0] 
offsets['light_thumb'] = struct.unpack('<I', in_file.read(4))[0] 
offsets['height_thumb'] = struct.unpack('<I', in_file.read(4))[0] 
offsets['assembly_info'] = struct.unpack('<I', in_file.read(4))[0] 
offsets['line_measure'] = struct.unpack('<I', in_file.read(4))[0] 
offsets['line_thickness'] = struct.unpack('<I', in_file.read(4))[0] 
offsets['string_data'] = struct.unpack('<I', in_file.read(4))[0]
# not sure if reserved is necessary offsets['reserved'] = struct.unpack('<I', in_file.read(4))[0]

log.debug("Exiting extract_offsets()")
return offsets

# color peak and color + light data extracted with extract_color_data
def extract_color_data(offset_dict, color_type, in_file):
    """extract_color_data
    Extracts RGB metadata and raw image data from a vk4 file. Stores data and
    returns as dictionary
    :param offset_dict: dictionary - offset values in vk4
    :param color_type: string - type of data, must be 'peak' or 'light'
    :param in_file: open file obj, must be vk4 file
    """ 
    log.debug("Entering extract_color_data()") 
    rgb_types = {'peak': 'color_peak', 'light': 'color_light'}
    rgb_color_data = dict() rgb_color_data['name'] = 'RGB ' + color_type
    in_file.seek(offset_dict[rgb_types[color_type]]) 
    rgb_color_data['width'] = struct.unpack('<I', in_file.read(4))[0] 
    rgb_color_data['height'] = struct.unpack('<I', in_file.read(4))[0] 
    rgb_color_data['bit_depth'] = struct.unpack('<I', in_file.read(4))[0] 
    rgb_color_data['compression'] = struct.unpack('<I', in_file.read(4))[0] 
    rgb_color_data['data_byte_size'] = struct.unpack('<I', in_file.read(4))[0] 
    rgb_color_arr = np.zeros(((rgb_color_data['width'] * rgb_color_data['height']), 
                              (rgb_color_data['bit_depth'] // 8)), dtype=np.uint8)
    i = 0 
    for val in range(rgb_color_data['width'] * rgb_color_data['height']):
        rgb = []
        for channel in range(3):
            rgb.append(ord(in_file.read(1)))
        rgb_color_arr[i] = rgb
        i = i + 1

    rgb_color_data['data'] = rgb_color_arr 
    log.debug("Exiting extract_color_data()")
    return rgb_color_data