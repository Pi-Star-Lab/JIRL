# Created by Sheelabhadra Dey
import cv2
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera

from config import CAMERA_HEIGHT, CAMERA_WIDTH, MAX_THROTTLE, MIN_THROTTLE
from config import STEERING_GAIN, STEERING_BIAS

class JetRacer(object):
    def __init__(self, camera_width=112, camera_height=112, fps=30):
        self.car = NvidiaRacecar()
        print("====== LOADING CAMERA ======")
        self.camera = CSICamera(width=camera_width, height=camera_height, capture_fps=fps)
        self.camera.running = True
        print("====== CAMERA LOADED SUCCESSFULLY ======")
        self.car.throttle = 0
        self.car.steering = 0

    def apply_throttle(self, throttle):
        self.car.throttle = float(throttle)

    def apply_steering(self, steer):
        self.car.steering = float(steer) * STEERING_GAIN + STEERING_BIAS

    def get_image(self):
        return self.camera.value.copy()

    def resize_image(self, image, width, height):
        return cv2.resize(image, (width, height))