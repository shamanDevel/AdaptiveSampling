import glob
import argparse
import os
import torch
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates histograms of volumes with .xyz, .raw and .cvol extensions.')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--density_minmax_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    torch.ops.load_library("./Renderer.dll")
    torch.ops.renderer.create_offscreen_context()
    torch.ops.renderer.initialize_renderer()
    
    volumes_paths = glob.glob(args.input_dir + "/*.xyz")
    volumes_paths += glob.glob(args.input_dir + "/*.raw")
    volumes_paths += glob.glob(args.input_dir + "/*.cvol")
    
    with open(args.density_minmax_file) as json_file:
        minmax_data = json.load(json_file)
    
    for volume_path in volumes_paths:
        _, volume_name = os.path.split(volume_path)
        _, extension = os.path.splitext(volume_path)
        
        if volume_name not in minmax_data:
            print("Min-Max density interval is not specified for ", volume_name)
            continue
        
        if extension == '.xyz':
            torch.ops.renderer.load_volume_from_xyz(volume_path)
        elif extension == ".raw":
            torch.ops.renderer.load_volume_from_raw(volume_path)
        elif extension == ".cvol":
            torch.ops.renderer.load_volume_from_binary(volume_path)
            
        ret = torch.ops.renderer.get_histogram()
        
        data = {}
        data['volume_name'] = volume_name
        data['interval_min'] = float(minmax_data[volume_name]["min"])
        data['interval_max'] = float(minmax_data[volume_name]["max"])
        data['volume_min'] = ret[0]
        data['volume_max'] = ret[1]
        raw_histogram = ret[2:]
        numof_bins = len(raw_histogram)
        data['resolution'] = (data['volume_max'] - data['volume_min']) / numof_bins
        data['histogram'] = raw_histogram[int(data['interval_min'] / data['resolution']):int(data['interval_max'] / data['resolution'])]

        with open(args.output_dir + "/" + volume_name.replace(".", "_") + "_hist.json", 'w') as out:
            json.dump(data, out)
        