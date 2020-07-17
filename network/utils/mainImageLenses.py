import math
import os
import numpy as np
import cv2 as cv
import imageio
import json

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import PIL as pl
from PIL import ImageTk, Image

if __name__ == "__main__":
    
    class GUI(tk.Tk):
        TARGET_WIDTH = 1000
        TARGET_HEIGHT = 1000*9//16

        def __init__(self):
            tk.Tk.__init__(self)
            self.title("Image Lenses")

            # members to be filled later
            self.pilImage = None
            self.tikImage = None
            self.inputImagePaths = []
            self.inputImages = []
            self.selectedImageIndex = 0
            self.selectedImage = None
            self.last_imags_dir = None
            self.last_settings_dir = None

            # root
            self.root_panel = tk.Frame(self)
            self.root_panel.pack(side="bottom", fill="both", expand="yes")
            self.panel = tk.Label(self.root_panel)
            self.emptyImage = np.zeros((GUI.TARGET_WIDTH, GUI.TARGET_HEIGHT, 3))
            self.setImage(self.emptyImage)
            self.panel.pack(side="left", fill="both", expand="yes")

            options = tk.Label(self.root_panel)

            # Input
            inputFrame = ttk.LabelFrame(options, text="Input", relief=tk.RIDGE)
            tk.Button(inputFrame, text="Load Images", command = lambda : self.loadImages()).pack()
            tk.Button(inputFrame, text="Remove all images", command = lambda : self.removeAllImages()).pack()
            self.inputImageListBox = tk.Listbox(inputFrame, selectmode=tk.SINGLE, width=30)
            self.inputImageListBox.bind("<Double-Button-1>", lambda _ : self.selectionChangeEvent())
            self.inputImageListBox.pack()
            inputFrame.pack()

            # Lense
            lenseFrame = ttk.LabelFrame(options, text="Lens", relief=tk.RIDGE)
            tk.Button(lenseFrame, text="Load Settings", command = lambda : self.loadLenseSettings()).pack()
            tk.Button(lenseFrame, text="Save Settings", command = lambda : self.saveLenseSettings()).pack()
            
            def setCropSize(e):
                self.cropSizeSlider.config(label="Crop size: %d"%(2**int(e)))
                self.updateImage()
            self.cropSizeSlider = tk.Scale(lenseFrame,
                                           from_=2, to=9,
                                           orient=tk.HORIZONTAL,
                                           resolution=1,
                                           showvalue=0,
                                           command=setCropSize,
                                           length=200)
            self.cropSizeSlider.set(4)
            self.cropSizeSlider.pack(anchor=tk.W, fill=tk.X)

            def setCropScale(e):
                self.cropScaleSlider.config(label="Crop scale: %d"%(int(e)))
                self.updateImage()
            self.cropScaleSlider = tk.Scale(lenseFrame,
                                           from_=2, to=16,
                                           orient=tk.HORIZONTAL,
                                           resolution=1,
                                           showvalue=0,
                                           command=setCropScale)
            self.cropScaleSlider.set(4)
            self.cropScaleSlider.pack(anchor=tk.W, fill=tk.X)

            def setLineSize(e):
                self.lineSizeSlider.config(label="Line size: %d"%(int(e)))
                self.updateImage()
            self.lineSizeSlider = tk.Scale(lenseFrame,
                                           from_=2, to=20,
                                           orient=tk.HORIZONTAL,
                                           resolution=1,
                                           showvalue=0,
                                           command=setLineSize)
            self.lineSizeSlider.set(5)
            self.lineSizeSlider.pack(anchor=tk.W, fill=tk.X)

            def setSourceX(e):
                self.sourceXSlider.config(label="Source X: %d"%(int(e)))
                self.updateImage()
            self.sourceXSlider = tk.Scale(lenseFrame,
                                           from_=0, to=1920,
                                           orient=tk.HORIZONTAL,
                                           resolution=1,
                                           showvalue=0,
                                           command=setSourceX)
            self.sourceXSlider.set(5)
            self.sourceXSlider.pack(anchor=tk.W, fill=tk.X)

            def setSourceY(e):
                self.sourceYSlider.config(label="Source Y: %d"%(int(e)))
                self.updateImage()
            self.sourceYSlider = tk.Scale(lenseFrame,
                                           from_=0, to=1080,
                                           orient=tk.HORIZONTAL,
                                           resolution=1,
                                           showvalue=0,
                                           command=setSourceY)
            self.sourceYSlider.set(5)
            self.sourceYSlider.pack(anchor=tk.W, fill=tk.X)

            def setTargetX(e):
                self.targetXSlider.config(label="Target X: %d"%(int(e)))
                self.updateImage()
            self.targetXSlider = tk.Scale(lenseFrame,
                                           from_=0, to=1920,
                                           orient=tk.HORIZONTAL,
                                           resolution=1,
                                           showvalue=0,
                                           command=setTargetX)
            self.targetXSlider.set(5)
            self.targetXSlider.pack(anchor=tk.W, fill=tk.X)

            def setTargetY(e):
                self.targetYSlider.config(label="Target Y: %d"%(int(e)))
                self.updateImage()
            self.targetYSlider = tk.Scale(lenseFrame,
                                           from_=0, to=1080,
                                           orient=tk.HORIZONTAL,
                                           resolution=1,
                                           showvalue=0,
                                           command=setTargetY)
            self.targetYSlider.set(500)
            self.targetYSlider.pack(anchor=tk.W, fill=tk.X)

            def setColor(e):
                self.colorSlider.config(label="Color: %d"%(int(e)))
                self.updateImage()
            self.colorSlider = tk.Scale(lenseFrame,
                                           from_=0, to=255,
                                           orient=tk.HORIZONTAL,
                                           resolution=1,
                                           showvalue=0,
                                           label='Color: 0',
                                           command=setColor)
            self.colorSlider.set(0)
            self.colorSlider.pack(anchor=tk.W, fill=tk.X)

            lenseFrame.pack()

            # Crop
            cropFrame = ttk.LabelFrame(options, text="Crop", relief=tk.RIDGE)
            tk.Button(cropFrame, text="Crop from current image", command = lambda : self.cropFromCurrent()).pack()
            cropValuesFrame = ttk.Frame(cropFrame)
            self.cropLeft = tk.Entry(cropValuesFrame, width=5)
            self.cropBottom = tk.Entry(cropValuesFrame, width=5)
            self.cropRight = tk.Entry(cropValuesFrame, width=5)
            self.cropTop = tk.Entry(cropValuesFrame, width=5)
            def updateCrop(e):
                self.updateImage()
            for i,entry in enumerate([self.cropLeft, self.cropBottom, self.cropRight, self.cropTop]):
                entry.bind("<Return>", updateCrop)
                entry.grid(row=0, column=i)
                entry.insert(0, "0")
            cropValuesFrame.pack()
            cropFrame.pack()

            # Export
            exportFrame = ttk.LabelFrame(options, text="Export", relief=tk.RIDGE)
            tk.Button(exportFrame, text="Export All", command = lambda : self.exportAllImages(False)).pack()
            tk.Button(exportFrame, text="Export All (as eps)", command = lambda : self.exportAllImages(True)).pack()
            exportFrame.pack()

            options.pack(side="left")

        def setImage(self, img):
            if not isinstance(img, pl.Image.Image):
                self.pilImage = pl.Image.fromarray(np.clip(img.transpose((1, 0, 2)), 0, 255).astype(np.uint8))
            else:
                self.pilImage = img
            self.pilImage = self.pilImage.resize((GUI.TARGET_WIDTH, GUI.TARGET_HEIGHT), pl.Image.BILINEAR)
            self.tikImage = ImageTk.PhotoImage(self.pilImage)
            self.panel.configure(image=self.tikImage)

        def loadImages(self):
            files = filedialog.askopenfilenames(
                parent = self.root_panel,
                initialdir="." if self.last_imags_dir is None else self.last_imags_dir,
                title = "Select input images",
                filetypes = (("PNG-Images", "*.png"), ("EPS-Images", "*.eps"), ("all files","*.*")) )
            if files is not None and len(files)>0:
                self.last_imags_dir = os.path.dirname(files[0])
                for file in files:
                    self.inputImagePaths.append(file)
                    self.inputImageListBox.insert(tk.END, os.path.basename(file))

        def removeAllImages(self):
            self.selectedImageIndex = -1
            self.selectedImage = None
            self.inputImagePaths = []
            self.inputImageListBox.delete(0, tk.END)
            self.setImage(self.emptyImage)

        def loadImage(self, imgPath):
            img = imageio.imread(imgPath)
            if len(img.shape)==2:
                # grayscale image
                img = np.stack([img]*3, axis=2)
            img = img.transpose((1, 0, 2))
            print(img.shape)
            print("min=", np.min(img), ", max=", np.max(img),
                    ", dtype=", img.dtype)
            if len(img.shape)==3 and img.shape[2]==4:
                # remove alpha -> blend to white
                oldtype = img.dtype
                newtype = np.float32
                white = 255*np.ones((img.shape[0], img.shape[1], 3), dtype=newtype)
                alpha = (img[:,:,3:4].astype(newtype)) / 255.0
                img = alpha*img[:,:,0:3].astype(newtype) + (1-alpha)*white
                img = img.astype(oldtype)
            return img

        def selectionChangeEvent(self):
            items = list(map(int, self.inputImageListBox.curselection()))
            if len(items)==0:
                print("nothing selected")
                self.selectedImageIndex = -1
                self.selectedImage = None
                self.setImage(self.emptyImage)
            else:
                self.selectedImageIndex = items[0]
                imgPath = self.inputImagePaths[self.selectedImageIndex]
                print("Selected:", imgPath)
                self.selectedImage = self.loadImage(imgPath)
                # process it and display it
                self.updateImage()

        def cropFromCurrent(self):
            if self.selectedImage is None:
                print("No image loaded")
                return
            width, height, _ = self.selectedImage.shape

            cropLeft = cropBottom = cropRight = cropTop = 0

            # fit image
            channel = -1
            while(cropLeft < width and np.all(self.selectedImage[cropLeft,:,channel]==255)):
                cropLeft += 1
            while(cropTop < height and np.all(self.selectedImage[:,cropTop,channel]==255)):
                cropTop += 1
            while(cropRight < width and np.all(self.selectedImage[width-cropRight-1,:,channel]==255)):
                cropRight += 1
            while(cropBottom < width and np.all(self.selectedImage[:, height-cropBottom-1, channel]==255)):
                cropBottom += 1

            # fit lens
            scale = self.cropScaleSlider.get()
            size = 2**self.cropSizeSlider.get()
            line = self.lineSizeSlider.get()
            cropLeft = max(0, min(
                cropLeft, 
                self.sourceXSlider.get()-line,
                self.targetXSlider.get()-line))
            cropTop = max(0, min(
                cropTop, 
                self.sourceYSlider.get()-line,
                self.targetYSlider.get()-line))
            cropRight = max(0, min(
                cropRight,
                width - (self.sourceXSlider.get()+size+line) - 1,
                width - (self.targetXSlider.get()+size*scale+line) - 1))
            cropBottom = max(0, min(
                cropBottom,
                height - (self.sourceYSlider.get()+size+line) - 1,
                height - (self.targetYSlider.get()+size*scale+line) - 1))

            # set text fields
            self.cropLeft.delete(0, tk.END);   self.cropLeft.insert(0, str(cropLeft))
            self.cropBottom.delete(0, tk.END); self.cropBottom.insert(0, str(cropBottom))
            self.cropRight.delete(0, tk.END);  self.cropRight.insert(0, str(cropRight))
            self.cropTop.delete(0, tk.END);    self.cropTop.insert(0, str(cropTop))
            self.updateImage()

        def loadLenseSettings(self):
            file = filedialog.askopenfilename(
                title="Load settings from .json",
                filetypes = (("Json file", "*.json"), ) )
            if file is None or not os.path.exists(file):
                print("No file selected")
                return
            with open(file, "r") as text_file:
                settings = json.load(text_file)
                print("Restore settings from", file)
                img = self.selectedImage #stop update on every .set call
                self.selectedImage = None
                self.cropSizeSlider.set(settings['cropSize'])
                self.cropScaleSlider.set(settings['cropScale'])
                self.lineSizeSlider.set(settings['lineSize'])
                self.sourceXSlider.set(settings['sourceX'])
                self.sourceYSlider.set(settings['sourceY'])
                self.targetXSlider.set(settings['targetX'])
                self.targetYSlider.set(settings['targetY'])
                self.colorSlider.set(settings.get('color', 0))
                self.cropLeft.delete(0, tk.END);   self.cropLeft.insert(0, settings.get('cropLeft', 0))
                self.cropBottom.delete(0, tk.END); self.cropBottom.insert(0, settings.get('cropBottom', 0))
                self.cropRight.delete(0, tk.END);  self.cropRight.insert(0, settings.get('cropRight', 0))
                self.cropTop.delete(0, tk.END);    self.cropTop.insert(0, settings.get('cropTop', 0))
                self.selectedImage = img # restore image and update
                self.updateImage()

        def saveLenseSettings(self):
            settings = {
                'cropSize' : int(self.cropSizeSlider.get()),
                'cropScale' : int(self.cropScaleSlider.get()),
                'lineSize' : int(self.lineSizeSlider.get()),
                'sourceX' : int(self.sourceXSlider.get()),
                'sourceY' : int(self.sourceYSlider.get()),
                'targetX' : int(self.targetXSlider.get()),
                'targetY' : int(self.targetYSlider.get()),
                'color' : int(self.colorSlider.get()),
                'cropLeft' : int(self.cropLeft.get()),
                'cropBottom' : int(self.cropBottom.get()),
                'cropRight' : int(self.cropRight.get()),
                'cropTop' : int(self.cropTop.get())
                }
            file = filedialog.asksaveasfilename(
                title="Save settings as .json",
                filetypes = (("Json file", "*.json"), ) )
            if file is not None:
                if not file.endswith(".json"):
                    file = file + ".json"
                with open(file, "w") as text_file:
                    text_file.write(json.dumps(settings, indent=2))
                print("Saved to", file)

        def processImage(self, imageIn : np.ndarray, hardCrop : bool):
            # duplicate image
            image = np.copy(imageIn)
            width, height, _ = image.shape
            print("width:", width, ", height:", height)
            # extract snippet
            scale = self.cropScaleSlider.get()
            size = 2**self.cropSizeSlider.get()
            color = (self.colorSlider.get(), self.colorSlider.get(), self.colorSlider.get())
            sourceStartX = self.sourceXSlider.get()
            sourceStartY = self.sourceYSlider.get()
            sourceEndX = min(width, self.sourceXSlider.get() + size)
            sourceEndY = min(height, self.sourceYSlider.get() + size)
            snippet = image[
                sourceStartX : sourceEndX,
                sourceStartY : sourceEndY,
                :]
            snippet = cv.resize(snippet, dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
            targetStartX = self.targetXSlider.get()
            targetStartY = self.targetYSlider.get()
            targetEndX = min(width, (self.targetXSlider.get() + size * scale))
            targetEndY = min(height, (self.targetYSlider.get() + size * scale))
            # paste snippet and draw borders
            def cvToNp(i):
                if isinstance(i, cv.UMat):
                    return i.get()
                else:
                    return i
            image = cvToNp(cv.rectangle(image,
                                 (sourceStartY, sourceStartX),
                                 (sourceEndY, sourceEndX),
                                 color,
                                 thickness = self.lineSizeSlider.get()))
            image[
                targetStartX : targetEndX,
                targetStartY : targetEndY,
                :] = snippet[:targetEndX-targetStartX, :targetEndY-targetStartY, :]
            image = cvToNp(cv.rectangle(image,
                                 (targetStartY, targetStartX),
                                 (targetEndY, targetEndX),
                                 color,
                                 thickness = self.lineSizeSlider.get()))
            # draw lines
            if (sourceStartY <= targetStartY and sourceStartX >= targetStartX) or \
                (sourceStartY >= targetStartY and sourceStartX <= targetStartX):
                image = cvToNp(cv.line(image, 
                                       (sourceStartY, sourceStartX),
                                       (targetStartY, targetStartX),
                                       color,
                                       thickness = self.lineSizeSlider.get(),
                                       lineType = cv.LINE_AA))
            if (sourceEndY >= targetEndY and sourceStartX >= targetStartX) or \
                (sourceEndY <= targetEndY and sourceStartX <= targetStartX) :
                image = cvToNp(cv.line(image, 
                                       (sourceEndY, sourceStartX),
                                       (targetEndY, targetStartX),
                                       color,
                                       thickness = self.lineSizeSlider.get(),
                                       lineType = cv.LINE_AA))
            if (sourceStartY <= targetStartY and sourceEndX <= targetEndX) or \
                (sourceStartY >= targetStartY and sourceEndX >= targetEndX):
                image = cvToNp(cv.line(image, 
                                       (sourceStartY, sourceEndX),
                                       (targetStartY, targetEndX),
                                       color,
                                       thickness = self.lineSizeSlider.get(),
                                       lineType = cv.LINE_AA))
            if (sourceEndY >= targetEndY and sourceEndX <= targetEndX) or \
                (sourceEndY <= targetEndY and sourceEndX >= targetEndX):
                image = cvToNp(cv.line(image, 
                                       (sourceEndY, sourceEndX),
                                       (targetEndY, targetEndX),
                                       color,
                                       thickness = self.lineSizeSlider.get(),
                                       lineType = cv.LINE_AA))

            # cropping
            cropLeft = int(self.cropLeft.get())
            cropBottom = int(self.cropBottom.get())
            cropRight = int(self.cropRight.get())
            cropTop = int(self.cropTop.get())
            if hardCrop:
                # change image size
                image = image[cropLeft:width-cropRight, cropTop:height-cropBottom, :]
            else:
                # only overlay with gray
                black = np.zeros_like(image)
                mask = np.ones_like(image) * 0.5
                mask[cropLeft:width-cropRight, cropTop:height-cropBottom, :].fill(1)
                t = image.dtype
                image = (mask * image + (1-mask) * black).astype(t)

            return image

        def updateImage(self):
            if self.selectedImage is None:
                return
            # process it
            result = self.processImage(self.selectedImage, False)
            # display it
            self.setImage(result)

        def exportAllImages(self, as_eps):
            for file in self.inputImagePaths:
                suffix = "." + file.rsplit('.', 1)[-1]
                prefix = file[:-len(suffix)]
                if as_eps:
                    suffix = ".eps" # force eps
                out_file = prefix + "_lens" + suffix

                print("Load", file)
                img = self.loadImage(file)
                img2 = self.processImage(img, True)
                img2 = img2.transpose((1, 0, 2))
                print("Save", out_file)
                imageio.imwrite(out_file, img2)

    gui = GUI()
    gui.mainloop()
