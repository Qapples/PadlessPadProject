import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


def main():
    image_labels = ["up", "down", "left", "right"]
    img_size = 128 * 128
    x_images = []
    y_labels = []

    for filename in os.listdir("training_images"):
        if not filename.endswith(".jpg"):
            continue

        label_parts = filename.split("_")[0:]
        labels = []

        for label in image_labels:
            labels.append(1 if label in label_parts else 0)

        image = cv2.imread(os.path.join("training_images", filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, img_size)

        x_images.append(image)
        y_labels.append(labels)

    x_images = np.array(x_images).astype('float32') / 255.0
    # x_images = x_images.reshape(num_frames, -1)
    y_labels = np.array(y_labels)

    model = Sequential()
    model.add(Dense(128, input_dim=6400, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    x_train = x_images[:80, :]
    y_train = np.random.randint(2, size=(80, 4))
    x_test = x_images[80:, :]
    y_test = np.random.randint(2, size=(20, 4))

    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=2)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test Loss: {loss:.3f}')
    print(f'Test Accuracy: {accuracy:.3f}')


if __name__ == "__main__":
    main()
