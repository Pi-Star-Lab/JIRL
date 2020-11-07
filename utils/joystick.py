import pygame
import os

class JoyStick(object):
    def __init__(self, index = 0):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.joystick.init()
        pygame.event.get()
        controller = pygame.joystick.Joystick(0)
        controller.init()
        self.controller = controller

    def _get_xbox_steer(self):
        return (self.controller.get_axis(0) - 0.078) / 2

    def _get_xbox_throttle(self):
        return (self.controller.get_axis(4) + 1) / 3

    def _get_logitech_driving_wheel_throttle(self):
        return (1 - self.controller.get_axis(2)) / 2

    def _get_logitech_driving_wheel_steer(self):
        return self.controller.get_axis(0)

    def _get_logitech_driving_wheel_brake(self):
         return (1 - self.controller.get_axis(3)) / 2 if \
             (1 - self.controller.get_axis(3)) / 2 > 0.01 else 0

    def get_throttle(self):
        pygame.event.get()
        return self.controller.get_axis(4)

    def get_steer(self):
        pygame.event.get()
        return -1 * self.controller.get_axis(0)

    def get_brake(self):
        return self._get_logitech_driving_wheel_brake()

    def is_stop(self):
        pygame.event.get()
        return self.controller.get_axis(5) > 0
    def is_on(self):
        pygame.event.get()
        return self.controller.get_axis(4) > 0.50



