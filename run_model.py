import os
import cv2
import keyboard
import numpy as np
from keras.models import load_model


def main():
    img_size = (128, 128)
    model_location = "model.h5"

    cap = cv2.VideoCapture(0)
    model = load_model(model_location)

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, img_size)

        frame_arr = np.array(frame).astype("float32") / 255.0
        frame_arr = frame_arr.flatten()
        frame_arr = frame_arr.reshape(1, frame_arr.shape[0])

        model_prediction = model.predict(frame_arr)
        print(f"Model prediction: {model_prediction}")

        cv2.imshow("frame", frame)

        if keyboard.is_pressed("q"):
            break

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
