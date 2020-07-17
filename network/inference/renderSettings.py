import numpy as np
import copy
import torch

class RenderSettings:
    """
    An instance of this class specifies all rendering settings.

    """

    def __init__(self):
        # constant per sample
        self.STEPSIZE = 0.1
        self.INTERPOLATION = 1
        self.RESOLUTION = [1024, 1024]
        self.VIEWPORT = [0,0,self.RESOLUTION[0],self.RESOLUTION[1]]
        self.TIMESTEPS = 10
        self.CAM_FOV = 45
        self.CAM_UP = [0,-1,0]
        self.CAM_ORIGIN_START = [0.5,0,0]
        self.CAM_LOOKAT_START = [0,0,0]
        self.CAM_ORIGIN_END = [0.5,0,0]
        self.CAM_LOOKAT_END = [0,0,0]
        self.MIPMAP_LEVEL = 0

        self.RENDER_MODE = 2 #0=iso, 2=dvr

        self.ISOVALUE_START = 0.3
        self.ISOVALUE_END = 0.3
        self.AO_SAMPLES = 256
        self.AO_RADIUS = 0.2

        self.OPACITY_SCALING= 40.0
        self.DENSITY_AXIS_OPACITY= []
        self.OPACITY_AXIS= []
        self.DENSITY_AXIS_COLOR= []
        self.COLOR_AXIS= []
        self.MIN_DENSITY= 0.0
        self.MAX_DENSITY= 1.0

        self.DVR_USE_SHADING = 1
        self.AMBIENT_LIGHT_COLOR = [0.1,0.1,0.1]
        self.DIFFUSE_LIGHT_COLOR = [0.9,0.9,0.9]
        self.SPECULAR_LIGHT_COLOR = [0.1,0.1,0.1]
        self.SPECULAR_EXPONENT = 16
        self.MATERIAL_COLOR = [1.0,1.0,1.0]
        self.LIGHT_DIRECTION = [0,0,1]
        self.VALUE_SCALING = 1.0

        # constant per timestep
        self.timestep = 0.0
        self.downsampling = 1

    def clone(self):
        return copy.deepcopy(self)

    def send(self):
        """
        Sends the settings to the renderer
        """
        set_params = torch.ops.renderer.set_renderer_parameter

        set_params("resolution", "%d,%d"%(self.RESOLUTION[0]//self.downsampling, self.RESOLUTION[1]//self.downsampling))
        set_params("viewport", "%d,%d,%d,%d"%(self.VIEWPORT[0]//self.downsampling, self.VIEWPORT[1]//self.downsampling,
                                              self.VIEWPORT[2]//self.downsampling, self.VIEWPORT[3]//self.downsampling))
        set_params("renderMode", "%d"%self.RENDER_MODE)
        set_params("aosamples", "%d"%self.AO_SAMPLES)
        set_params("aoradius", "%f"%self.AO_RADIUS)
        set_params("cameraFoV", "%f"%self.CAM_FOV)
        set_params("cameraUp", "%f,%f,%f"%tuple(self.CAM_UP))
        set_params("stepsize", "%f"%self.STEPSIZE)
        set_params("interpolation", "%d"%self.INTERPOLATION)
        set_params("mipmapLevel", "%d"%self.MIPMAP_LEVEL)

        set_params("opacityScaling", "%f"%self.OPACITY_SCALING)
        set_params("densityAxisOpacity", ",".join(map(str, self.DENSITY_AXIS_OPACITY)))
        set_params("opacityAxis", ",".join(map(str, self.OPACITY_AXIS)))
        set_params("densityAxisColor", ",".join(map(str, self.DENSITY_AXIS_COLOR)))
        set_params("colorAxis", ",".join([str(val) for vals in self.COLOR_AXIS for val in vals]))
        set_params("minDensity", "%f"%self.MIN_DENSITY)
        set_params("maxDensity", "%f"%self.MAX_DENSITY)

        set_params("dvrUseShading", "%d"%self.DVR_USE_SHADING)
        set_params("ambientLightColor", "%f,%f,%f"%tuple(self.AMBIENT_LIGHT_COLOR))
        set_params("diffuseLightColor", "%f,%f,%f"%tuple(self.DIFFUSE_LIGHT_COLOR))
        set_params("specularLightColor", "%f,%f,%f"%tuple(self.SPECULAR_LIGHT_COLOR))
        set_params("specularExponent", "%d"%self.SPECULAR_EXPONENT)
        set_params("materialColor", "%f,%f,%f"%tuple(self.MATERIAL_COLOR))
        set_params("lightDirection", "%f,%f,%f"%tuple(self.LIGHT_DIRECTION))

        def lerp(a, b, t):
            return a * (1-t) + b * t
        camOriginStart = np.array(self.CAM_ORIGIN_START)
        camOriginEnd   = np.array(self.CAM_ORIGIN_END)
        camTargetStart = np.array(self.CAM_LOOKAT_START)
        camTargetEnd   = np.array(self.CAM_LOOKAT_END)
        isoStart = self.ISOVALUE_START
        isoEnd = self.ISOVALUE_END

        t = self.timestep
        set_params("cameraOrigin", "%f,%f,%f"%tuple(lerp(camOriginStart, camOriginEnd, t).tolist()))
        set_params("cameraLookAt", "%f,%f,%f"%tuple(lerp(camTargetStart, camTargetEnd, t).tolist()))
        set_params("isovalue", "%f"%lerp(isoStart, isoEnd, t))

    def update_camera(self, camera, end = False):
        """
        Updates this instance with the settings from the specified inference.Camera instance.
        end=False: update the start values
        end=True: update the end values
        """
        self.RESOLUTION[0] = camera.resX
        self.RESOLUTION[1] = camera.resY
        self.VIEWPORT = [0,0,self.RESOLUTION[0],self.RESOLUTION[1]]
        if end:
            self.CAM_LOOKAT_END = camera.getLookAt()
            self.CAM_ORIGIN_END = camera.getOrigin()
        else:
            self.CAM_LOOKAT_START = camera.getLookAt()
            self.CAM_ORIGIN_START = camera.getOrigin()
        self.CAM_UP = camera.getUp()

    def to_dict(self):
        """
        Converts the settings into a dictionary to be saved in a json.
        """
        d = {}
        for key, value in self.__dict__.items():
            if key.isupper():
                d[key] = value
        return d

    def from_dict(self, d):
        """
        Fills this instance with the specified dictionary loaded from a json
        """
        for key, value in self.__dict__.items():
            if key.isupper() and key in d:
                self.__dict__[key] = d[key]