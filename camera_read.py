import keyboard
import time
import sys
import cv2
import os

output_directory = "training_images"
labels = ['up', 'down', 'left', 'right']
current_keys_pressed = []
break_flag = False

if not os.path.exists(output_directory):
	os.makedirs(output_directory)

def save_data(frame, img_id):
    filename = f'{img_id}_{"_".join(current_keys_pressed)}.jpg'
    filepath = os.path.join(output_directory, filename)
    cv2.imwrite(filepath, frame)

def on_key_press(key):
	if key.name == "q":
		break_flag = True
	elif key in labels:
		current_keys_pressed.append(key.name)	

def on_key_release(key):
	if key.name == "q":
		break_flag = True
	elif key in labels:
		current_keys_pressed.remove(key.name)

keyboard.on_press(on_key_press)
keyboard.on_release(on_key_release)

cap = cv2.VideoCapture(0)
capture_rate_hz = 50 #in hZ
image_count = 0

while True:
	if break_flag:
		break

	ret, frame = cap.read()
	cv2.imshow('frame', frame)

	save_data(frame, image_count)
	
	cv2.waitKey(20)
	image_count += 1
	
cap.release()
cv2.destroyAllWindows()
