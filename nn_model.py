import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

cap = cv2.VideoCapture('video.mp4')

num_frames = 100

frames = []

for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (80, 80))
    frames.append(frame)

data = np.array(frames).astype('float32') / 255.0
data = data.reshape(num_frames, -1)

model = Sequential()
model.add(Dense(128, input_dim=6400, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

x_train = data[:80,:]
y_train = np.random.randint(2, size=(80, 4))
x_test = data[80:,:]
y_test = np.random.randint(2, size=(20, 4))

model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=2)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Loss: {loss:.3f}')
print(f'Test Accuracy: {accuracy:.3f}')
