import argparse
import math
import os
import os.path
import subprocess
import time

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import PIL as pl
from PIL import ImageTk, Image
from tkcolorpicker import askcolor

from utils import ScreenSpaceShading

INITIAL_DIR = "D:\\VolumeSuperResolution-InputData"

class GUI(tk.Tk):

    CHANNEL_MASK = 1
    CHANNEL_NORMAL = 2
    CHANNEL_DEPTH = 3
    CHANNEL_AO = 4
    CHANNEL_COLOR = 5
    CHANNEL_FLOW = 6

    def __init__(self):
        tk.Tk.__init__(self)
        self.title('Dataset Viewer')
        self.input_name = None

        # members to be filled later
        self.pilImage = None
        self.tikImage = None
        self.num_frames = 0
        self.selected_entry = 0
        self.selected_time = 0
        self.last_folder = None
        self.dataset_file = None
        self.hdf5_file = None
        self.dset_keys =[]
        self.dset = None
        self.mode = None

        # root
        self.root_panel = tk.Frame(self)
        self.root_panel.pack(side="bottom", fill="both", expand="yes")
        self.panel = tk.Label(self.root_panel)
        self.black = np.zeros((512, 512, 3))
        self.setImage(self.black)
        self.panel.pack(side="left", fill="both", expand="yes")

        options1 = tk.Label(self.root_panel)

        # Input
        inputFrame = ttk.LabelFrame(options1, text="Input", relief=tk.RIDGE)
        tk.Button(inputFrame, text="Open HDF5", command=lambda : self.openHDF5()).pack()
        listbox_frame = tk.Frame(inputFrame)
        self.dset_listbox_scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        self.dset_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, yscrollcommand=self.dset_listbox_scrollbar.set)
        self.dset_listbox_scrollbar.config(command=self.dset_listbox.yview)
        self.dset_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dset_listbox.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=1)
        self.dset_listbox.bind("<Double-Button-1>", self.setDsetCallback)
        listbox_frame.pack(anchor=tk.W, fill=tk.X)
        self.selected_dset = None
        self.dataset_entry_slider = tk.Scale(
            inputFrame,
            from_=0, to=0,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="Entry",
            showvalue=0,
            command=lambda e: self.setEntry(int(e)))
        self.dataset_entry_slider.pack(anchor=tk.W, fill=tk.X)
        self.dataset_time = 0
        self.dataset_time_slider = tk.Scale(
            inputFrame,
            from_=0, to=0,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="Time",
            showvalue=0,
            command=lambda e: self.setTime(int(e)))
        self.dataset_time_slider.pack(anchor=tk.W, fill=tk.X)
        inputFrame.pack(fill=tk.X)

        # Channel
        channelsFrame = ttk.LabelFrame(options1, text="Channel", relief=tk.RIDGE)
        self.channel_mode = tk.IntVar()
        self.channel_mode.set(GUI.CHANNEL_COLOR)
        self.channel_mode.trace_add('write', lambda a,b,c : self.updateImage())
        tk.Radiobutton(channelsFrame, text="Mask", variable=self.channel_mode, value=GUI.CHANNEL_MASK).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="Normal", variable=self.channel_mode, value=GUI.CHANNEL_NORMAL).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="Depth", variable=self.channel_mode, value=GUI.CHANNEL_DEPTH).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="AO", variable=self.channel_mode, value=GUI.CHANNEL_AO).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="Color", variable=self.channel_mode, value=GUI.CHANNEL_COLOR).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="Flow", variable=self.channel_mode, value=GUI.CHANNEL_FLOW).pack(anchor=tk.W)
        channelsFrame.pack(fill=tk.X)

        # Shading
        self.shading = ScreenSpaceShading('cpu')
        self.shading.fov(30)
        self.ambient_light_color = np.array([0.1,0.1,0.1])
        self.shading.ambient_light_color(self.ambient_light_color)
        self.diffuse_light_color = np.array([0.8, 0.8, 0.8])
        self.shading.diffuse_light_color(self.diffuse_light_color)
        self.specular_light_color = np.array([0.02, 0.02, 0.02])
        self.shading.specular_light_color(self.specular_light_color)
        self.shading.specular_exponent(16)
        self.shading.light_direction(np.array([0.1,0.1,1.0]))
        self.material_color = np.array([1.0,1.0,1.0])#[1.0, 0.3, 0.0])
        self.shading.material_color(self.material_color)
        self.shading.ambient_occlusion(0.5)
        self.shading.background(np.array([1.0, 1.0, 1.0]))

        # Save
        tk.Button(options1, text="Save Image", command=lambda : self.saveImage()).pack()
        self.saveFolder = "/"

        options1.pack(side="left")

    def openHDF5(self):
        dataset_file = filedialog.askopenfilename(
            initialdir=self.last_folder,
            title = "Select HDF5 file",
            filetypes = (("HDF5 files", "*.hdf5"),))
        print(dataset_file)
        if dataset_file is not None:
            self.dataset_file = str(dataset_file)
            print("New hdf5 selected:", self.dataset_file)
            self.title(self.dataset_file)
            self.last_folder = os.path.dirname(self.dataset_file)
            # load hdf5 file
            self.hdf5_file = h5py.File(dataset_file, "r")
            # list datasets
            self.dset_listbox.delete(0, tk.END)
            self.dset_keys = list(self.hdf5_file.keys())
            for key in self.dset_keys:
                self.dset_listbox.insert(tk.END, key)
            self.dset_listbox.selection_set(first=0)
            self.setDset(self.dset_keys[0])

        else:
            print("No folder selected")

    def setDsetCallback(self, *args):
        items = self.dset_listbox.curselection()
        items = [self.dset_keys[int(item)] for item in items]
        self.setDset(items[0])

    def setDset(self, name):
        self.selected_dset = name
        print("Select dataset '%s'"%(self.selected_dset))
        self.dset = self.hdf5_file[self.selected_dset]
        self.mode = self.dset.attrs.get("Mode", "IsoUnshaded")
        print("Mode:", self.mode)
        # find number of entries
        entries = self.dset.shape[0]
        num_frames = self.dset.shape[1]
        print("Number of entries found:", entries, "with", num_frames, "timesteps")
        print("Image size:", self.dset.shape[3], "x", self.dset.shape[4])
        self.entries = entries
        self.num_frames = num_frames
        self.dataset_entry_slider.config(to=entries-1)
        self.dataset_time_slider.config(to=num_frames-1)
        self.setEntry(self.selected_entry if self.selected_entry<entries else 0)

    def setEntry(self, entry):
        entry = min(entry, self.entries-1)
        self.dataset_entry_slider.config(label='Entry: %d'%int(entry))
        self.selected_entry = entry
        self.updateImage()

    def setTime(self, entry):
        entry = min(entry, self.num_frames-1)
        self.dataset_time_slider.config(label='Time: %d'%int(entry))
        self.selected_time = entry
        self.updateImage()

    def setImage(self, img):
        if img.dtype == np.uint8:
            self.pilImage = pl.Image.fromarray(img.transpose((1,0,2)))
        else:
            self.pilImage = pl.Image.fromarray(np.clip((img*255).transpose((1, 0, 2)), 0, 255).astype(np.uint8))
        if self.pilImage.size[0] <= 256:
            self.pilImage = self.pilImage.resize((self.pilImage.size[0]*2, self.pilImage.size[1]*2), pl.Image.NEAREST)
        self.tikImage = ImageTk.PhotoImage(self.pilImage)
        self.panel.configure(image=self.tikImage)

    def saveImage(self):
        filename =  filedialog.asksaveasfilename(initialdir = self.saveFolder,title = "Save as",filetypes = (("png files", "*.png"),("jpeg files","*.jpg"),("all files","*.*")))
        if len(filename)==0:
            return;
        if len(os.path.splitext(filename)[1])==0:
            filename = filename + ".png"
        self.pilImage.save(filename)
        self.saveFolder = os.path.dirname(filename)

    def updateImage(self):
        if self.dset is None:
            # no image loaded
            self.setImage(self.black)
            return

        def selectChannel(img):
            if self.mode == "DVR":
                mask = img[3:4,:,:]
                if self.channel_mode.get() == GUI.CHANNEL_MASK:
                    return np.concatenate((mask, mask, mask))
                elif self.channel_mode.get() == GUI.CHANNEL_COLOR:
                    return img[0:3,:,:]
                elif self.channel_mode.get() == GUI.CHANNEL_NORMAL:
                    return (img[4:7,:,:] * 0.5 + 0.5) * mask
                elif self.channel_mode.get() == GUI.CHANNEL_DEPTH:
                    return np.concatenate([img[7:8,:,:]]*3, axis=0)
                elif self.channel_mode.get() == GUI.CHANNEL_FLOW:
                    return np.concatenate((img[8:9,:,:]*10, img[9:10,:,:]*10, np.zeros_like(img[6:7,:,:]))) + 0.5
                else:
                    return self.black

            else: # IsoUnshaded
                if self.channel_mode.get() == GUI.CHANNEL_MASK:
                    if img.dtype == np.uint8:
                        mask = img[0:1,:,:]
                    else:
                        mask = img[0:1,:,:] * 0.5 + 0.5
                    return np.concatenate((mask, mask, mask))
                elif self.channel_mode.get() == GUI.CHANNEL_NORMAL:
                    if img.dtype == np.uint8:
                        return img[1:4,:,:]
                    else:
                        return img[1:4,:,:] * 0.5 + 0.5
                elif self.channel_mode.get() == GUI.CHANNEL_DEPTH:
                    return np.concatenate((img[4:5,:,:], img[4:5,:,:], img[4:5,:,:]))
                elif self.channel_mode.get() == GUI.CHANNEL_AO:
                    return np.concatenate((img[5:6,:,:], img[5:6,:,:], img[5:6,:,:]))
                elif self.channel_mode.get() == GUI.CHANNEL_FLOW:
                    return np.concatenate((img[6:7,:,:]*10, img[7:8,:,:]*10, np.zeros_like(img[6:7,:,:]))) + 0.5
                elif self.channel_mode.get() == GUI.CHANNEL_COLOR:
                    if img.dtype == np.uint8:
                        shading_input = torch.unsqueeze(torch.from_numpy(img.astype(np.float32)/255.0), 0)
                        shading_input[:, 1:4, :, :] = shading_input[:, 1:4, :, :] * 2 - 1
                    else:
                        shading_input = torch.unsqueeze(torch.from_numpy(img), 0)
                    shading_output = self.shading(shading_input)[0]
                    return torch.clamp(shading_output, 0, 1).cpu().numpy()

        img = self.dset[self.selected_entry, self.selected_time, :,:,:]
        img = selectChannel(img)
        img = img.transpose((2, 1, 0))
        self.setImage(img)

if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()
