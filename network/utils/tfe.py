import tkinter as tk
from tkinter import filedialog
import PIL
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import json
import os

class LabColorInterpolation:
    def __init__(self, rgbA, rgbB):
        import skimage.color
        col = np.array([[list(rgbA), list(rgbB)]], dtype='uint8')
        lab = skimage.color.rgb2lab(col)
        self.labA = lab[0,0]
        self.labB = lab[0,1]
    def interpolate(self, alpha):
        import skimage.color
        lab = (1-alpha)*self.labA + alpha*self.labB
        lab = [[lab]]
        rgb = skimage.color.lab2rgb(lab) * 255
        return rgb[0,0]

class TransferFunctionEditor(object):
    """description of class"""
    
    MIN_WIDTH = 400
    COLOR_HEIGHT = 50
    ALPHA_HEIGHT = 100
    MARKER_SIZE = 6

    colorPoints = []
    alphaPoints = []

    selection = 0
    clicked = False
    idCounter = 1

    def __init__(self, parent, callback = None):
        """
        Transfer function editor
         parent: parent window
         callback: function that is called for every change with 'this' as parameter
        """
        self.callback = callback
        self.container = tk.Label(parent)
        
        self.containerL = tk.Label(self.container)
        self.colorCanvas = tk.Label(self.containerL, width=TransferFunctionEditor.MIN_WIDTH, height=TransferFunctionEditor.COLOR_HEIGHT)
        self.colorCanvas.pack(side="top", fill="both", expand=True)
        self.alphaCanvas = tk.Label(self.containerL, width=TransferFunctionEditor.MIN_WIDTH, height=TransferFunctionEditor.ALPHA_HEIGHT)
        self.alphaCanvas.pack(side="top", fill="both", expand=True)
        self.containerL.pack(side="left", fill="both")

        self.colorCanvas.bind("<ButtonPress-1>", lambda e: self._OnPress_Color(e, 1))
        self.colorCanvas.bind("<Double-ButtonPress-1>", lambda e: self._OnPress_Color(e, 2))
        self.colorCanvas.bind("<ButtonPress-3>", lambda e: self._OnRightPress_Color(e))
        self.colorCanvas.bind("<ButtonRelease-1>", self._OnRelease)
        self.colorCanvas.bind("<B1-Motion>", self._OnMotion_Color)

        self.alphaCanvas.bind("<ButtonPress-1>", lambda e: self._OnPress_Alpha(e, 1))
        self.alphaCanvas.bind("<Double-ButtonPress-1>", lambda e: self._OnPress_Alpha(e, 2))
        self.alphaCanvas.bind("<ButtonPress-3>", lambda e: self._OnRightPress_Alpha(e))
        self.alphaCanvas.bind("<ButtonRelease-1>", self._OnRelease)
        self.alphaCanvas.bind("<B1-Motion>", self._OnMotion_Alpha)

        self.colorImage = Image.new(mode="RGB", size=(TransferFunctionEditor.MIN_WIDTH, TransferFunctionEditor.COLOR_HEIGHT))
        self.alphaImage = Image.new(mode="RGB", size=(TransferFunctionEditor.MIN_WIDTH, TransferFunctionEditor.ALPHA_HEIGHT))

        self.containerR = tk.Label(self.container)
        self.newButtonIcon = tk.PhotoImage(file="utils/iconNewFile.png").subsample(2)
        self.newButton = tk.Button(self.containerR, image=self.newButtonIcon, command=lambda : self._clear())
        self.newButton.pack()
        self.loadButtonIcon = tk.PhotoImage(file="utils/iconLoad.png").subsample(2)
        self.loadButton = tk.Button(self.containerR, image=self.loadButtonIcon, command=lambda : self._open())
        self.loadButton.pack()
        self._saveButtonIcon = tk.PhotoImage(file="utils/iconSave.png").subsample(2)
        self._saveButton = tk.Button(self.containerR, image=self._saveButtonIcon, command=lambda : self._save())
        self._saveButton.pack()
        self.containerR.pack()

        self.container.pack(fill="both", expand=True)
        self._clear()
        self._redrawColor()
        self._redrawAlpha()

        self.lastDir = "."

    def _callCallback(self):
        if self.callback is not None:
            self.callback(self)

    def computeTransferFunction(self, resolution : int):
        """
        Computes the transfer function.
        Returns a numpy array of shape (resolution, 4) containing the RGBA-values as float32
        """
        tf = np.empty((resolution, 4), dtype=np.float32)
        w = resolution

        # color
        self.colorPoints.sort(key=lambda x: x[0])
        tf[0:int(self.colorPoints[0][0]*w), 0:3] = np.array([[c / 255.0 for c in self.colorPoints[0][1]]])
        for i in range(1,len(self.colorPoints)):
            x1,c1,id1 = self.colorPoints[i-1]
            x2,c2,id2 = self.colorPoints[i]
            interp = LabColorInterpolation(c1, c2)
            for x in range(int(x1*w), int(x2*w)):
                f = (x - int(x1*w)) / (int(x2*w)-int(x1*w))
                c = interp.interpolate(f)
                tf[x,0:3] = c / 255
        tf[int(self.colorPoints[-1][0]*w):, 0:3] = np.array([[c / 255.0 for c in self.colorPoints[-1][1]]])

        # alpha
        self.alphaPoints.sort(key=lambda x: x[0])
        tf[0:int(self.alphaPoints[0][0]*w), 3] = np.array([self.alphaPoints[0][1]])
        for i in range(1, len(self.alphaPoints)):
            x1,c1,id1 = self.alphaPoints[i-1]
            x2,c2,id2 = self.alphaPoints[i]
            for x in range(int(x1*w), int(x2*w)):
                f = (x - int(x1*w)) / (int(x2*w)-int(x1*w))
                a = (1-f)*c1 + f*c2
                tf[x,3] = a
        tf[int(self.alphaPoints[-1][0]*w):, 3] = np.array([self.alphaPoints[-1][1]])

        return tf

    def _nextID(self):
        self.idCounter += 1
        return self.idCounter

    def _redrawColor(self):
        d = ImageDraw.Draw(self.colorImage)

        self.colorPoints.sort(key=lambda x: x[0])
        w,h = self.colorImage.size

        # draw gradient
        d.rectangle([0,0,self.colorPoints[0][0]*w,h], fill=self.colorPoints[0][1])
        for i in range(1,len(self.colorPoints)):
            x1,c1,id1 = self.colorPoints[i-1]
            x2,c2,id2 = self.colorPoints[i]
            interp = LabColorInterpolation(c1, c2)
            for x in range(int(x1*w), int(x2*w)):
                f = (x - int(x1*w)) / (int(x2*w)-int(x1*w))
                c = tuple(int(v) for v in interp.interpolate(f))
                #c = tuple(int((1-f)*e1+f*e2) for e1,e2 in zip(c1,c2))
                d.rectangle([x,0,x+1,h], fill=c)
        d.rectangle([self.colorPoints[-1][0]*w,0,w,h], fill=self.colorPoints[-1][1])

        # draw control points
        ms = TransferFunctionEditor.MARKER_SIZE
        for i in range(len(self.colorPoints)):
            x,c,id = self.colorPoints[i]
            x = int(x*w)
            fill = 0 if self.selection==id else 150
            d.ellipse([x-ms-1,0,x+ms,h], fill=(fill, fill, fill))
            d.ellipse([x-4,3,x+3,h-3], fill=c)

        self.tikColorImage = ImageTk.PhotoImage(self.colorImage)
        self.colorCanvas.configure(image=self.tikColorImage)

    def _redrawAlpha(self):
        d = ImageDraw.Draw(self.alphaImage)

        self.alphaPoints.sort(key=lambda x: x[0])
        w,h = self.alphaImage.size
        d.rectangle([0,0,w,h], fill=(0,0,0))

        # draw lines
        x1,c1,id1 = self.alphaPoints[0]
        d.line([0, h-c1*h-1, x1*w, h-c1*h-1], fill=(200,200,200))
        for i in range(1, len(self.alphaPoints)):
            x1,c1,id1 = self.alphaPoints[i-1]
            x2,c2,id2 = self.alphaPoints[i]
            d.line([x1*w, h-c1*h-1, x2*w, h-c2*h-1], fill=(200,200,200))
        x1,c1,id1 = self.alphaPoints[-1]
        d.line([x1*w, h-c1*h-1, w, h-c1*h-1], fill=(200,200,200))

        # draw control points
        ms = TransferFunctionEditor.MARKER_SIZE / 2
        for i in range(len(self.alphaPoints)):
            x,c,id = self.alphaPoints[i]
            fill = 250 if self.selection==id else 200
            outline = (250,50,50) if self.selection==id else None
            d.ellipse([x*w-ms,h-c*h-1-ms, x*w+ms, h-c*h-1+ms], 
                      fill=(fill, fill, fill), outline=outline)

        self.tikAlphaImage = ImageTk.PhotoImage(self.alphaImage)
        self.alphaCanvas.configure(image=self.tikAlphaImage)

    def _OnPress_Color(self, event, count):
        #print("Mouse Pressed - color, x:", event.x, ", y:", event.y, ", count:", count)
        self.clicked = True
        has_changes = False

        w,h = self.colorImage.size
        self.selection = 0
        ms = TransferFunctionEditor.MARKER_SIZE
        for i in range(len(self.colorPoints)-1, -1, -1):
            x,c,id = self.colorPoints[i]
            x = int(x*w)
            if (x-event.x)**2/(ms*ms) + (h/2-event.y)**2/(h*h/4) <= 1:
                self.selection = id
                break

        if count==2:
            if self.selection > 0:
                # _open color picker
                from tkcolorpicker import askcolor
                for i in range(len(self.colorPoints)):
                    x,c,id = self.colorPoints[i]
                    if id==self.selection:
                        newColor = askcolor(c)[0]
                        if newColor is not None:
                            self.colorPoints[i] = (x,newColor, id)
                            has_changes = True
                        break
            elif self.selection == 0:
                # add new point
                # fetch color at that position
                newColor = self.colorImage.getpixel((event.x, h//2))
                newX = event.x / w
                newID = self._nextID()
                self.colorPoints.append((newX, newColor, newID))
                #print("Point at",newX,"added with color",newColor)
                self.selection = newID
                has_changes = True

        self._redrawColor()
        if has_changes:
            self._callCallback()

    def _OnPress_Alpha(self, event, count):
        #print("Mouse Pressed - alpha, x:", event.x, ", y:", event.y, ", count:", count)
        self.clicked = True
        has_changes = False

        w,h = self.alphaImage.size
        self.selection = 0
        ms = TransferFunctionEditor.MARKER_SIZE
        for i in range(len(self.alphaPoints)-1, -1, -1):
            x,c,id = self.alphaPoints[i]
            x = int(x*w)
            y = int(h-c*h-1)
            if (x-event.x)**2/(ms*ms) + (y-event.y)**2/(ms*ms) <= 1:
                self.selection = id
                break

        if count==2:
            if self.selection == 0:
                # add new point
                # fetch alpha at that position
                newX = event.x / w
                newY = 1 - event.y / h
                newID = self._nextID()
                self.alphaPoints.append((newX, newY, newID))
                #print("Point at",newX,"added with value",newY)
                self.selection = newID
                has_changes = True

        self._redrawAlpha()
        if has_changes:
            self._callCallback()

    def _OnRightPress_Color(self, event):
        #print("Mouse right pressed - color, x:", event.x, ", y:", event.y)
        self.clicked = True
        has_changes = False

        w,h = self.colorImage.size
        self.selection = 0
        ms = TransferFunctionEditor.MARKER_SIZE
        for i in range(len(self.colorPoints)-1, -1, -1):
            x,c,id = self.colorPoints[i]
            x = int(x*w)
            if (x-event.x-2)**2/(ms*ms) + (h/2-event.y)**2/(h*h/4) <= 1:
                self.selection = id
                break

        if self.selection > 0 and len(self.colorPoints)>1:
            # delete that item
            self.colorPoints = [(x,c,id) for x,c,id in self.colorPoints if id!=self.selection]
            self.selection = 0
            has_changes = True

        self._redrawColor()
        if has_changes:
            self._callCallback()

    def _OnRightPress_Alpha(self, event):
        #print("Mouse right pressed - alpha, x:", event.x, ", y:", event.y)
        self.clicked = True
        has_changes = False

        w,h = self.alphaImage.size
        self.selection = 0
        ms = TransferFunctionEditor.MARKER_SIZE
        for i in range(len(self.alphaPoints)-1, -1, -1):
            x,c,id = self.alphaPoints[i]
            x = int(x*w)
            y = int(h-c*h-1)
            if (x-event.x)**2/(ms*ms) + (y-event.y)**2/(ms*ms) <= 1:
                self.selection = id
                break

        if self.selection > 0 and len(self.alphaPoints)>1:
            # delete that item
            self.alphaPoints = [(x,c,id) for x,c,id in self.alphaPoints if id!=self.selection]
            self.selection = 0
            has_changes = True

        self._redrawAlpha()
        if has_changes:
            self._callCallback()

    def _OnRelease(self, event):
        self.clicked=False

    def _OnMotion_Color(self, event):
        #print("Mouse Motion, x:", event.x, ", selection:", self.selection)
        w,h = self.colorImage.size
        if self.selection>0 and self.clicked:
            for i in range(len(self.colorPoints)):
                x,c,id = self.colorPoints[i]
                if id==self.selection:
                    newX = max(0.0, min(1.0, event.x/w))
                    self.colorPoints[i] = (newX, c, id)
                    #print("Point moved to ", newX)
                    self._redrawColor()
                    self._callCallback()
                    return

    def _OnMotion_Alpha(self, event):
        #print("Mouse Motion, x:", event.x, ", selection:", self.selection)
        w,h = self.alphaImage.size
        if self.selection>0 and self.clicked:
            for i in range(len(self.alphaPoints)):
                x,c,id = self.alphaPoints[i]
                if id==self.selection:
                    newX = max(0.0, min(1.0, event.x/w))
                    newY = max(0.0, min(1.0, 1 - event.y/h))
                    self.alphaPoints[i] = (newX, newY, id)
                    #print("Point moved to ", newX, newY)
                    self._redrawAlpha()
                    self._callCallback()
                    return

    def _clear(self):
        #add basic points
        self.colorPoints = [(0.0, (255,0,0), self._nextID()), (1.0, (255,255,255), self._nextID())]
        self.alphaPoints = [(0.0, 0.0, self._nextID()), (1.0, 1.0, self._nextID())]
        self._callCallback()

    INPUT_FILETYPES = (("Transfer Function", "*.json"),)
    def _open(self):
        # _open file
        file = tk.filedialog.askopenfilename(
            initialdir="." if self.lastDir is None else self.lastDir,
            title="Select transfer function file",
            filetypes = TransferFunctionEditor.INPUT_FILETYPES)
        if file is None or not os.path.exists(file):
            return
        self.lastDir = os.path.dirname(file)
        
        oldCounter = self.idCounter
        try: 
            # read json
            with open(file, "r") as f:
                data = json.load(f)
            # extract data
            self.idCounter = 0
            colorPoints = [(v[0], tuple(v[1]), self._nextID()) for v in data['color']]
            alphaPoints = [(v[0], v[1], self._nextID()) for v in data['alpha']]
            self.colorPoints = colorPoints
            self.alphaPoints = alphaPoints
        except KeyboardInterrupt:
            raise
        except:
            self.idCounter = oldCounter
            print("Unable to load transfer function:", sys.exc_info()[0])
            return

        self._redrawColor()
        self._redrawAlpha()
        print("Transfer function loaded from", file)
        self._callCallback()

    def _save(self):
        #assemble json
        self.colorPoints.sort(key=lambda x: x[0])
        self.alphaPoints.sort(key=lambda x: x[0])
        colors = [(v[0], v[1]) for v in self.colorPoints]
        alpha = [(v[0], v[1]) for v in self.alphaPoints]
        data = {"color": colors, "alpha":alpha}
        # select file
        file = tk.filedialog.asksaveasfilename(
            initialdir="." if self.lastDir is None else self.lastDir,
            title="Select transfer function file",
            filetypes = TransferFunctionEditor.INPUT_FILETYPES)
        if file is None or file=="":
            return
        self.lastDir = os.path.dirname(file)
        if not file.endswith(".json"):
            file = file + ".json"
        # _save json
        with open(file, "w") as f:
            json.dump(data, f)
        print("Saved to", file)

if __name__=="__main__":
    
    class TransferFunctionTestGui(tk.Tk):
        def __init__(self):
            tk.Tk.__init__(self)
            self.title("Transfer Function Editor")
            self.resizable(False, False)
            self.root_panel = tk.Frame(self)
            self.root_panel.pack(side="bottom", fill="both", expand="yes")

            def callback(tfe : TransferFunctionEditor):
                tf = tfe.computeTransferFunction(256)
                #print(tf)

            self.tfe = TransferFunctionEditor(self.root_panel, callback)

    gui = TransferFunctionTestGui()
    gui.mainloop()