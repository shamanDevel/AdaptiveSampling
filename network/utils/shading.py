import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ScreenSpaceShading(nn.Module):
    """
    Performs screen space shading.
    
    Input to the shading (aka output of the superresolution network):
    Tensor of shape B x C x H x W with C=7
      channel 0,1,2: rgb albedo
      channel 3,4,5: screen space normal
      channel 6: mask

    Parameters:
     - ambient light color
     - diffuse light color
     - specular light color
     - specular exponent
     - screen space light direction
    """

    def __init__(self, device):
        super().__init__()
        self._device = device
        self.enable_specular = True
        self.inverse_ao = False # If set to true, AO will be inverted (1-AO)
        self._background = torch.from_numpy(np.array([0,0,0])).to(device=self._device, dtype=torch.float32).view((1,3,1,1))

    def fov(self, fov):
        """Field of view in degrees"""
        assert isinstance(fov, (float, int))
        assert fov > 0 and fov < 90, "fov has to be in (0,90)"
        self._fov = float(fov)
        self._eyedirs = dict()
        return self

    def get_fov(self):
        return self._fov
    
    def ambient_light_color(self, color):
        """Ambient light color"""
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        self._ambient_light_color = torch.from_numpy(color).to(device=self._device, dtype=torch.float32).view((1,3,1,1))
        return self

    def diffuse_light_color(self, color):
        """Diffuse light color"""
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        self._diffuse_light_color = torch.from_numpy(color).to(device=self._device, dtype=torch.float32).view((1,3,1,1))
        return self

    def specular_light_color(self, color):
        """Specular light color"""
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        self._specular_light_color = torch.from_numpy(color).to(device=self._device, dtype=torch.float32).view((1,3,1,1))
        return self

    def specular_exponent(self, exponent):
        """The specular exponent"""
        assert isinstance(exponent, (int,float))
        if isinstance(exponent, float):
            assert exponent.is_integer()
            exponent = int(exponent)
        assert exponent > 0
        self._specular_exponent = exponent

    def light_direction(self, dir):
        """Screen-space light direction"""
        assert isinstance(dir, np.ndarray)
        assert dir.shape == (3,)
        self._light_direction = torch.from_numpy(dir / np.linalg.norm(dir)).to(device=self._device, dtype=torch.float32).view((1,3,1,1))
        return self

    def material_color(self, color):
        """Material color"""
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        self._material_color = torch.from_numpy(color).to(device=self._device, dtype=torch.float32).view((1,3,1,1))
        return self

    def ambient_occlusion(self, ao):
        """Sets the ambient occlusion strength"""
        self._ao = float(ao)
        return self

    def background(self, color):
        """Background color"""
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        self._background = torch.from_numpy(color).to(device=self._device, dtype=torch.float32).view((1,3,1,1))
        return self

    def _get_eyedir(self, h, w):
        if (h, w) in self._eyedirs:
            return self._eyedirs[(h, w)]

        aspect = w / h
        fov_radians = math.radians(self._fov * 10)
        f = math.tan(fov_radians/2)
        z_near = 0.1
        z_far = 1.0

        a_11 = 1/(f*aspect)
        a_22 = 1/f
        a_33 = -(z_near + z_far)/(z_far - z_near)
        a_34 = -2*z_near*z_far/(z_far - z_near)

        perspective_matrix = np.array([
            [a_11, 0, 0, 0],       
            [0, a_22, 0, 0],       
            [0, 0, a_33, a_34],    
            [0, 0, -1, 0]          
        ]).T

        # TODO: speed up this computation
        # TODO: I think, it is not completely correct yet
        print('Compute eye rays: ',(w, h))
        #eyedir = np.zeros((3, h, w), dtype=np.float32)
        #for y in range(h):
        #    for x in range(w):
        #        v1 = np.array([[x/float(w)*2-1, -(y/float(h)*2-1), z_far, 1]]).T
        #        v1 = np.matmul(perspective_matrix, v1)
        #        v1 /= v1[3]
        #        v2 = np.array([[x/float(w)*2-1, -(y/float(h)*2-1), z_near, 1]]).T
        #        v2 = np.matmul(perspective_matrix, v2)
        #        v2 /= v2[3]
        #        v = v2 - v1
        #        v = v[0:3,0] / np.linalg.norm(v[0:3,0])
        #        eyedir[:,y,x] = np.array([0,0,1])#v
        #print('eyedir[0,0]:', eyedir[:,0,0])
        #print('eyedir[h,0]:', eyedir[:,-1,0])
        #print('eyedir[0,w]:', eyedir[:,0,-1])
        #print('eyedir[h,w]:', eyedir[:,-1,-1])
        #print('eyedir[center]:', eyedir[:,h//2,w//2])
        eyedir = np.array([0.0,0.0,1.0], dtype=np.float32)
        eyedir = np.repeat(np.repeat(eyedir[:,np.newaxis,np.newaxis], h, axis=1), w, axis=2)
        eyedir = torch.from_numpy(eyedir).to(self._device)
        self._eyedirs[(h, w)] = eyedir
        return eyedir

    #@profile
    def forward(self, input):
        B,C,H,W = input.shape
        assert C>=5
        input_mask = input[:,0:1,:,:]
        input_normal = ScreenSpaceShading.normalize(input[:,1:4,:,:], dim=1)
        input_depth = input[:,4:5,:,:]
        if C>=6:
            if self.inverse_ao:
                input_ao = self._ao * torch.clamp(1.0-input[:,5:6,:,:], 0, 1) \
                    + (1-self._ao) * torch.ones_like(input[:,5:6,:,:])
            else:
                input_ao = self._ao * torch.clamp(input[:,5:6,:,:], 0, 1) \
                        + (1-self._ao) * torch.ones_like(input[:,5:6,:,:])
        else:
            input_ao = torch.ones_like(input[:,4:5,:,:])

        color = torch.zeros((B,3,H,W), dtype=torch.float32, device=self._device)

        # ambient color
        color += self._ambient_light_color * self._material_color

        # diffuse color
        diffuse_factor = torch.abs(torch.sum(self._light_direction * input_normal, dim=1, keepdim=True))
        diffuse_color = self._diffuse_light_color * self._material_color
        color += diffuse_color * diffuse_factor

        # specular color
        if self.enable_specular:
            eyedir = self._get_eyedir(H, W)
            reflect = 2 * torch.sum(self._light_direction * input_normal, dim=1, keepdim=True) * input_normal - self._light_direction
            specular_factor = ((self._specular_exponent + 2) / (2 * np.pi)) * \
                (torch.clamp(torch.sum(reflect * eyedir, dim=1, keepdim=True), 0, 1) ** self._specular_exponent)
            color += specular_factor * self._specular_light_color # no material color

        # ambient occlusion
        color *= input_ao

        # masking
        def lerp(a, b, x):
            return a + x * (b - a)
        color = lerp(self._background, color, torch.clamp(input_mask*0.5+0.5, 0, 1))

        # done
        return torch.clamp(color, 0, 1)

    @staticmethod
    def normalize(input, dim):
        """
        Normalizes the row specified by the given dimension.
        For each row in the tensor at the specified image, the row is treated as a vector
         and the vector is normalized (v <- v / ||v||).
        All other dimensions are treated as batches.

        This method is save against vectors of length zero.
        """

        EPSILON = 1e-7
        epsilon = torch.ones(1, dtype=input.dtype, device=input.device) * EPSILON
        lengths = torch.max(torch.norm(input, dim=dim, keepdim=True), epsilon)
        return input / lengths
        
