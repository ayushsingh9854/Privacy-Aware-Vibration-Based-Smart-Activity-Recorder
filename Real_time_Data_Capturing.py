import serial
import numpy as np
import joblib

model = joblib.load("E:\study\code\MINOR_PROJECT\model.pkl")

ser = serial.Serial("COM13", 9600)  

WINDOW_SIZE = 100
window = []

while True:
    try:
        line = ser.readline().decode().strip()
        x, y, z = map(float, line.split(","))

        mag = (x**2 + y**2 + z**2)**0.5
        window.append(mag)

        if len(window) == WINDOW_SIZE:
            data = np.array(window)
            data = data - data.mean()

            features = [
                data.mean(),
                data.std(),
                data.max(),
                data.min(),
                np.sqrt(np.mean(data**2))
            ]

            pred = model.predict([features])
            print("Prediction:", pred[0])

            window = []

    except:
        continue