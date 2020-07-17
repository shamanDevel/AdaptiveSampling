from xml.dom import minidom
import glob
import numpy as np
from skimage import color
from scipy import stats
import argparse
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os
import json

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
    
def load_tf_v2(path):
    with open(path) as json_file:
        tf = json.load(json_file)
        
    tf["densityAxisOpacity"] = [str(val) for val in tf["densityAxisOpacity"]]
    tf["opacityAxis"] = [str(val) for val in tf["opacityAxis"]]
    tf["densityAxisColor"] = [str(val) for val in tf["densityAxisColor"]]
    tf["colorAxis"] = [str(val) for vals in tf["colorAxis"] for val in vals]
    
    return tf
    
def fit_gmm(histogram, numof_components):
    x_axis = np.asarray([i / len(histogram) for i in range(len(histogram))])
    
    resolution = 2.0 ** (-14.0)
    repetition_per_element = np.asarray([int(i / resolution) for i in histogram])
    data = np.repeat(x_axis, repetition_per_element)
    
    gmm = GaussianMixture(n_components = numof_components)
    gmm = gmm.fit(X=data.reshape(-1, 1))
    gmm_y = np.exp(gmm.score_samples(x_axis.reshape(-1, 1)))
    
    center_values = np.exp(gmm.score_samples(np.asarray(gmm.means_)))
    
    centers = [x[0] for _,x in sorted(zip(center_values, gmm.means_), reverse=True)]
    
    return x_axis, gmm_y, centers
    
def get_best_fit_gmm(histogram, min_numof_components):
    x_axis = np.asarray([i / len(histogram) for i in range(len(histogram))])
    
    resolution = 2.0 ** (-14.0)
    repetition_per_element = np.asarray([int(i / resolution) for i in histogram])
    data = np.repeat(x_axis, repetition_per_element)
    
    best_gmm = None
    best_score = float("inf")
    for i in range(min_numof_components, 16):
        gmm = GaussianMixture(n_components = i).fit(X=data.reshape(-1, 1))
        
        bic = gmm.bic(data.reshape(-1, 1))
        if bic < best_score:
            best_gmm = gmm
            best_score = bic
            
    gmm_y = np.exp(best_gmm.score_samples(x_axis.reshape(-1, 1)))
    center_values = np.exp(best_gmm.score_samples(np.asarray(best_gmm.means_)))
    center_probabilities = center_values / np.sum(center_values)
    
    #center_probabilities = best_gmm.weights_ / np.sum(best_gmm.weights_)
    
    return x_axis, gmm_y, center_probabilities, best_gmm.means_
    
def show_fitted_gaussian(histogram, numof_components):
    x_axis, gmm_y, _ = fit_gmm(histogram, numof_components)

    # Plot histograms and gaussian curves
    fig, ax = plt.subplots()
    ax.plot(x_axis, histogram)
    ax.plot(x_axis, gmm_y / np.max(gmm_y) * np.max(np.asarray(histogram)))

    plt.legend()
    plt.show()
    
def show_best_fitted_gaussian(histogram, min_numof_components):
    x_axis, gmm_y, _, _ = get_best_fit_gmm(histogram, min_numof_components)

    # Plot histograms and gaussian curves
    fig, ax = plt.subplots()
    ax.plot(x_axis, histogram)
    ax.plot(x_axis, gmm_y / np.max(gmm_y) * np.max(np.asarray(histogram)))

    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates transfer functions from Scivis Xml colormap files.')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--histogram_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--numof_tfs_per_volume', type=str, required=True)
    parser.add_argument('--min_peak_width', type=float, default=0.005)
    parser.add_argument('--max_peak_width', type=float, default=0.030)
    parser.add_argument('--min_num_peaks', type=int, default=3)
    parser.add_argument('--max_num_peaks', type=int, default=5)
    parser.add_argument('--min_peak_height', type=float, default=0.1)
    parser.add_argument('--max_peak_height', type=float, default=1.0)
    args = parser.parse_args()

    colormaps = glob.glob(args.input_dir + "/*.xml")
    histograms = glob.glob(args.histogram_dir + "/*_hist.json")

    min_numof_opacity_cps = args.min_num_peaks
    max_numof_opacity_cps = args.max_num_peaks

    number_of_tfs = int(args.numof_tfs_per_volume)
    numof_decimals = int(math.floor(math.log10(number_of_tfs - 1)) + 1.0)
    for histogram_path in histograms:
        with open(histogram_path) as json_file:
            histogram_info = json.load(json_file)
            
        _, _, center_probabilities, gmm_centers = get_best_fit_gmm(histogram_info["histogram"], max_numof_opacity_cps)
        sampler = stats.rv_discrete(name='sampler', values=(np.arange(len(center_probabilities)), center_probabilities))
        print("Number of components used: ", len(gmm_centers))
        
        #show_best_fitted_gaussian(histogram_info["histogram"], max_numof_opacity_cps)
        #show_fitted_gaussian(histogram_info["histogram"], 15)
            
        for tf_number in range(number_of_tfs):
            chosen_idx = np.random.randint(len(colormaps))
            density_axis_color, color_axis = parse_xml(colormaps[chosen_idx])

            numof_opacity_cps = np.random.randint(min_numof_opacity_cps, max_numof_opacity_cps + 1)
            sampled = sampler.rvs(size=1024)
            peak_centers_indices = list(dict.fromkeys(sampled))[0:numof_opacity_cps]
            peak_centers = sorted(gmm_centers[peak_centers_indices])
            peak_centers = [x[0] for x in peak_centers]
            
            # Sample opacity density and opacity values.
            density_axis_opacity = []
            opacity_axis = []
            prev = 0.0
            for idx in range(numof_opacity_cps):
                center = peak_centers[idx]
                next_center = 1.0 if idx == (numof_opacity_cps - 1) else peak_centers[idx + 1]
                
                peak_width = args.min_peak_width + np.random.rand() * (args.max_peak_width - args.min_peak_width)
                peak_width = min(peak_width, center - prev, (next_center - center) / 2)
                left_density = center - peak_width
                right_density = 1.0 if idx == (numof_opacity_cps - 1) and np.random.rand() < 0.5 else center + peak_width

                center_opacity = args.min_peak_height + np.random.rand() * (args.max_peak_height - args.min_peak_height)
                
                density_axis_opacity += [left_density, center, right_density]
                opacity_axis += [0.0, center_opacity, 0.0 if right_density < 1.0 else center_opacity]
                
                prev = right_density
            
            #Save
            tf_folder_histogram = histogram_info["volume_name"]
            tf_folder_histogram = args.output_dir + "/" + tf_folder_histogram[0:tf_folder_histogram.rfind('.')] + "/"
        
            if not os.path.exists(tf_folder_histogram):
                os.makedirs(tf_folder_histogram)
            
            with open(tf_folder_histogram + ("{:0" + str(numof_decimals) + "d}").format(tf_number) + ".tf", 'w') as out:
                data = {}
                data["densityAxisOpacity"] = density_axis_opacity
                data["opacityAxis"] = opacity_axis
                data["densityAxisColor"] = density_axis_color
                data["colorAxis"] = np.asarray(color_axis).reshape(-1, 3).tolist()
                data["minDensity"] = histogram_info["interval_min"]
                data["maxDensity"] = histogram_info["interval_max"]
                json.dump(data, out)
