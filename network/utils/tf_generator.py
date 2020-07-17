from xml.dom import minidom
import glob
import numpy as np
from skimage import color
import argparse
import math

def parse_xml(path):
    colormap_xml = minidom.parse(path)
    items = colormap_xml.getElementsByTagName("Point")
    
    density_axis_color = []
    color_axis = []
    for item in items:
        density_axis_color += [float(item.attributes["x"].value)]
        r = float(item.attributes["r"].value)
        g = float(item.attributes["g"].value)
        b = float(item.attributes["b"].value)
        
        lab = color.rgb2lab([[[r, g, b]]])
        color_axis += [(lab[0][0][0], lab[0][0][1], lab[0][0][2])]
        
    return density_axis_color, color_axis
    
def save_tf(path, density_axis_opacity, opacity_axis, density_axis_color, color_axis):
    with open(path, 'w') as tf_file:
        tf_file.write(str(len(density_axis_opacity)) + " ")
        for density in density_axis_opacity:
            tf_file.write(str(density) + " ")
    
        for opacity in opacity_axis:
            tf_file.write(str(opacity) + " ")
    
        tf_file.write(str(len(density_axis_color)) + " ")
        for density in density_axis_color:
            tf_file.write(str(density) + " ")
    
        for color in color_axis:
            tf_file.write(str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + " ")
            
def load_tf(path):
    with open(path, 'r') as tf_file:
        line = tf_file.readline()
        numbers = line.split(" ")
        
        numof_opacity_cps = int(numbers[0])
        density_axis_opacity = numbers[1:numof_opacity_cps + 1]
        opacity_axis = numbers[numof_opacity_cps + 1:2 * numof_opacity_cps + 1]
        
        numof_color_cps = int(numbers[2 * numof_opacity_cps + 1])
        density_axis_color = numbers[2 * numof_opacity_cps + 2:2 * numof_opacity_cps + 2 + numof_color_cps]
        color_axis = numbers[2 * numof_opacity_cps + 2 + numof_color_cps:-1]
    
    return density_axis_opacity, opacity_axis, density_axis_color, color_axis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates transfer functions from Scivis Xml colormap files.')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--numof_tfs', type=str, required=True)
    parser.add_argument('--min_peak_width', type=float, default=0.005)
    parser.add_argument('--max_peak_width', type=float, default=0.02)
    parser.add_argument('--min_num_peaks', type=int, default=3)
    parser.add_argument('--max_num_peaks', type=int, default=5)
    parser.add_argument('--min_peak_height', type=float, default=0.1)
    parser.add_argument('--max_peak_height', type=float, default=1.0)
    args = parser.parse_args()

    colormaps = glob.glob(args.input_dir + "/*.xml")

    min_numof_opacity_cps = args.min_num_peaks
    max_numof_opacity_cps = args.max_num_peaks
    number_of_tfs = int(args.numof_tfs)
    numof_decimals = int(math.floor(math.log10(number_of_tfs - 1)) + 1.0)
    leftmost_density = 0.05
    for tf_number in range(number_of_tfs):
        chosen_idx = np.random.randint(len(colormaps))
        density_axis_color, color_axis = parse_xml(colormaps[chosen_idx])
        density_axis_color_to_sample = [density for density in density_axis_color if density > leftmost_density]

        numof_opacity_cps = np.random.randint(min_numof_opacity_cps, np.minimum(len(density_axis_color_to_sample), max_numof_opacity_cps))
        peak_centers = sorted(np.random.choice(density_axis_color_to_sample, numof_opacity_cps, replace=False))
        
        # Sample opacity density and opacity values.
        density_axis_opacity = []
        opacity_axis = []
        prev = 0.0
        predefined_bottom_range = 0.07
        for idx in range(numof_opacity_cps):
            center = peak_centers[idx]
            next_center = 1.0 if idx == (numof_opacity_cps - 1) else peak_centers[idx + 1]
            
            #bottom_range_left = np.minimum(predefined_bottom_range, (center - prev) * args.peak_width)
            #bottom_range_right = np.minimum(predefined_bottom_range, (next_center - center) * args.peak_width)
            #left_density = center - np.random.uniform(0.0, bottom_range_left)
            #right_density = center + np.random.uniform(0.0, bottom_range_right)
            
            peak_width = args.min_peak_width + np.random.rand() * (args.max_peak_width-args.min_peak_width)
            peak_width = min(peak_width, center-prev, (next_center-prev)/2)
            left_density = center - peak_width
            right_density = center + peak_width

            #center_opacity = np.maximum(0.0, np.minimum(1.0, np.random.normal(0.8, 0.07)))
            center_opacity = args.min_peak_height + np.random.rand() * (args.max_peak_height - args.min_peak_height)
            
            density_axis_opacity += [left_density, center, right_density]
            opacity_axis += [0.0, center_opacity, 0.0]
            
            prev = right_density
        
        save_tf(args.output_dir + "/" + ("{:0" + str(numof_decimals) + "d}").format(tf_number) + ".tf", density_axis_opacity, opacity_axis, density_axis_color, color_axis)
