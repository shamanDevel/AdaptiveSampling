import torch
import argparse
import os.path
from tkinter import filedialog, Tk

print("Import library")
torch.ops.load_library("../bin/Renderer.dll")

def convert(input_filename, output_filename):
    print("Load", input_filename)
    if torch.ops.renderer.load_volume_from_raw(input_filename) != 1:
        print("Failed to load")
        return
    if torch.ops.renderer.save_volume_to_binary(output_filename) != 1:
        print("Failed to save")
        return
    print("Converted")

if __name__== "__main__":
    parser = argparse.ArgumentParser(
        description="Converts raw volumes described by a .dat file into our custom .cvol format")
    parser.add_argument("input", help="Input file, a .dat file", nargs='?', default=None)
    parser.add_argument("output", help="Output file, a .cvol file", nargs='?', default=None)
    args = parser.parse_args()

    folderMode = 0

    Tk().withdraw()
    if args.input is None:
        file = filedialog.askopenfilename(
            title="Select input checkpoint",
            filetypes = (("Raw Volume", "*.dat"), ) )
        if file is None or not os.path.exists(file):
            print("No input file selected")
            exit(-1)
        input_filename = file
    else:
        if os.path.isdir(args.input):
            folderMode += 1
        elif not args.input.endswith(".dat"):
            print("Selected input file is not a .dat file")
            exit(-1)
        input_filename = args.input
    print("Selected input file:", input_filename)

    if args.output is None:
        file = filedialog.asksaveasfilename(
            title="Select output script file",
            filetypes = (("Binary Volume", "*.cvol"), ) )
        output_filename = file
    else:
        if os.path.isdir(args.output):
            folderMode += 1
        elif not args.output.endswith(".pt"):
            print("Selected output file is not a .cvol file")
            exit(-1)
        output_filename = args.output
    print("Selected output file:", output_filename)

    if folderMode == 2:
        print("Batch convert")
        #files = [f for f in os.listdir(input_filename) if f.endswith(".dat")]
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(input_filename):
            for file in f:
                if '.dat' in file:
                    files.append(os.path.join(r, file))
        for f in files:
            input = os.path.join(input_filename, f)
            output = os.path.join(output_filename, os.path.basename(f)[:-4]+".cvol")
            if os.path.exists(output):
                print("Skipping", output)
            else:
                convert(input, output)
    elif folderMode == 0:
        convert(input_filename, output_filename)
    else:
        print("Either specify input and output files or input and output directories")
