import keras
import pandas as pd
import numpy as np
import os
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
import time
import vae
from vae.controller import VAEController
from stable_baselines import SAC
car = NvidiaRacecar()
#path = "jetcar_weights.pkl"
path = "logs/sac/JetVae-v0_43/JetVae-v0.pkl"
input_array = np.zeros((1, 256))
try:
    i = 0

    v = VAEController()
    v.load("logs/vae-256.pkl")
    #model = keras.models.load_model(path)
    model = SAC.load(path)
    camera = CSICamera(width=112, height=112)
    camera.running = True

    print("Imported Camera! Ready to Start!")
    while True:
        print("Starting to read image")
        image = camera.value.copy()
        print("Image Read!")
        #image = cv2.resize(image, (224//2, 224//2))
        print(type(image))
        tmp = v.encode_from_raw_image(image)
        print("Got image")
        input_array[0,:] = tmp
        print("Predicing from Model")
        action, _ = model.predict(input_array, deterministic = True)
        t = (action[1] + 1) / 2
        # Convert from [0, 1] to [min, max]
        action[1] = (1 - t) * 0.45 + 0.6 * t

        print(action)
        steer = float(action[0])
        throttle = float(action[1])
        print("Frame: {}\t Steer: {}\t Throttle:{}".format(i, steer, throttle))
        i+=1
        car.throttle = throttle
        car.steering = steer 
except:
    car.throttle = 0
    car.steer = 0
    import sys
    sys.exit(1)
