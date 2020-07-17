import math
from enum import Enum

OrientationNames = [
	"Xp", "Xm", "Yp", "Ym", "Zp", "Zm"
]
OrientationUp = [
	[1,0,0], [-1,0,0],
	[0,1,0], [0,-1,0],
	[0,0,1], [0,0,-1]
]
OrientationPermutation = [
	[ 2,-1,-3], [-2, 1, 3],
	[ 1, 2, 3], [-1,-2,-3],
	[-3,-1, 2], [ 3, 1,-2]
]
OrientationInvertYaw = [
	True, False, False, True, False, True
]
OrientationInvertPitch = [
	False, False, False, False, False, False
]

class Camera:
    def __init__(self, resX, resY, origin = [0, 1, -1.7], orientation = 2):
        self.resX = resX
        self.resY = resY
        self.lookAt = [0, 0, 0]
        self.speed = 0.01
        self.zoomspeed = 1.1
        self.orientation = orientation

        self.currentDistance, self.currentPitch, self.currentYaw = Camera.toAngles(origin)
        self.baseDistance = self.currentDistance
        self.zoomvalue = 0

    @staticmethod
    def toAngles(pos):
        length = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        pitch = math.asin(pos[1] / length)
        yaw = math.atan2(pos[2], pos[0])
        return length, pitch, yaw
    @staticmethod
    def fromAngles(length, pitch, yaw):
        pos = [0,0,0]
        pos[1] = math.sin(pitch) * length
        pos[0] = math.cos(pitch) * math.cos(yaw) * length
        pos[2] = math.cos(pitch) * math.sin(yaw) * length
        return pos

    def getLookAt(self):
        return self.lookAt

    def getOrigin(self):
        o1 = Camera.fromAngles(self.currentDistance, 
                               self.currentPitch * (+1 if OrientationInvertPitch[self.orientation] else -1), 
                               self.currentYaw * (+1 if OrientationInvertYaw[self.orientation] else -1))
        o2 = [None]*3
        for i in range(3):
            p = OrientationPermutation[self.orientation][i]
            o2[i] = o1[abs(p)-1] * (1 if p>0 else -1)
        for i in range(3): o2[i] += self.lookAt[i]
        return o2

    def getUp(self):
        return OrientationUp[self.orientation]

    def startMove(self):
        self.oldDistance = self.currentDistance
        self.oldPitch = self.currentPitch
        self.oldYaw = self.currentYaw

    def stopMove(self):
        pass

    def move(self, deltax, deltay):
        self.currentPitch = max(math.radians(-80), min(math.radians(80), self.oldPitch + self.speed * deltay))
        self.currentYaw = self.oldYaw + self.speed * deltax
        #print("pitch:", self.currentPitch, ", yaw:", self.currentYaw)

    def zoom(self, delta):
        self.zoomvalue += delta
        self.currentDistance = self.baseDistance * (self.zoomspeed ** self.zoomvalue)
        #print("dist:", self.currentDistance)

    def from_dict(self, d : dict):
        self.currentPitch = math.radians(float(d["currentPitch"]))
        self.currentYaw = math.radians(float(d["currentYaw"]))
        self.zoomvalue = float(d["zoomValue"])
        self.zoomspeed = float(d["zoomSpeed"])
        self.lookAt[0] = float(d["lookAt"][0])
        self.lookAt[1] = float(d["lookAt"][1])
        self.lookAt[2] = float(d["lookAt"][2])
        self.orientation = int(d["orientation"])
        self.baseDistance = 1.0
        self.currentDistance = self.baseDistance * (self.zoomspeed ** self.zoomvalue)
