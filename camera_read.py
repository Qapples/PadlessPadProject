import keyboard
import time
import sys
import cv2
import os


def main():
    output_directory = "training_images"
    labels = ["up", "down", "left", "right"]
    no_label_key = "space"
    current_keys_pressed = []
    break_flag = False

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def save_data(frame, img_id):
        filename = f'{img_id}_{"_".join(current_keys_pressed)}.jpg' if len(current_keys_pressed) > 0 else f"{img_id}.jpg"
        filepath = os.path.join(output_directory, filename)
        cv2.imwrite(filepath, frame)

        print(f"saved to {filepath}")

    cap = cv2.VideoCapture(0)
    image_count = 0

    while True:
        if break_flag:
            break

        ret, frame = cap.read()
        cv2.imshow("frame", frame)

        for key in labels:
            if keyboard.is_pressed(key):
                current_keys_pressed.append(key)

        if len(current_keys_pressed) > 0 or keyboard.is_pressed(no_label_key):
            save_data(frame, image_count)

        current_keys_pressed.clear()
        cv2.waitKey(20)
        image_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
