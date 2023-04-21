import cv2
import os

# Define the four labels
labels = ['up', 'down', 'left', 'right']

# Define the directory paths to save the labeled data
data_dirs = [os.path.join(os.getcwd(), label) for label in labels]

# Create the directories if they don't exist
for dir_path in data_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Define the function to save the labeled data
def save_data(frame, label):
    data_dir = data_dirs[label]
    filename = f'{label}_{len(os.listdir(data_dir))}.jpg'
    filepath = os.path.join(data_dir, filename)
    cv2.imwrite(filepath, frame)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('frame', frame)

    # Wait for a keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Check if the 'q' key was pressed to exit the program
    if key == ord('q'):
        break

    # Check if a label key was pressed
    for i, label in enumerate(labels):
        if key == ord(str(i)):
            save_data(frame, i)
            print(f'{label} saved')

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
