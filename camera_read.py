import cv2
import os

labels = ['up', 'down', 'left', 'right']

data_dirs = [os.path.join(os.getcwd(), label) for label in labels]

for dir_path in data_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_data(frame, label):
    data_dir = data_dirs[label]
    filename = f'{label}_{len(os.listdir(data_dir))}.jpg'
    filepath = os.path.join(data_dir, filename)
    cv2.imwrite(filepath, frame)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    for i, label in enumerate(labels):
        if key == ord(str(i)):
            save_data(frame, i)
            print(f'{label} saved')

cap.release()
cv2.destroyAllWindows()
