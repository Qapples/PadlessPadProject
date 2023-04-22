import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
from sklearn.model_selection import KFold


def main():
    image_labels = ["up", "down", "left", "right"]
    img_size = (128, 128)
    x_images = []
    y_labels = []

    k_fold_splits = 5
    epochs = 10
    batch_size = 10

    model_save_path = "model.h5"

    img_size_1d = 1
    for val in img_size:
        img_size_1d *= val

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
    x_images = np.reshape(x_images, (-1, img_size_1d))
    y_labels = np.array(y_labels)

    def build_model(input_size):
        out_model = Sequential()
        out_model.add(Input(shape=(input_size,)))
        out_model.add(Flatten())
        out_model.add(Dense(256, activation='relu'))
        out_model.add(Dense(128, activation='relu'))
        out_model.add(Dense(64, activation='relu'))
        out_model.add(Dense(4, activation='sigmoid'))

        out_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

        return out_model

    model = build_model(img_size[0] * img_size[1])

    loss_arr = []
    acc_arr = []
    k_fold = KFold(n_splits=k_fold_splits, shuffle=True)
    for fold_num, (train_idx, test_idx) in enumerate(k_fold.split(x_images)):
        print(f"------- Training Fold#{fold_num + 1} Begin -------\n")

        x_train, y_train = x_images[train_idx], y_labels[train_idx]
        x_test, y_test = x_images[test_idx], y_labels[test_idx]

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"\nLoss: {loss:.3f}. Accuracy: {accuracy:.3f}")

        loss_arr.append(loss)
        acc_arr.append(accuracy)

        print(f"------- Training Fold#{fold_num + 1} End -------\n")

    print("-------- Average results across folds --------")
    print(f"Avg loss: {np.mean(loss_arr)} +- {np.std(loss_arr)}")
    print(f"Avg acc: {np.mean(acc_arr)} +- {np.std(acc_arr)}")

    print(f"\nSaving model to {model_save_path}")
    model.save(model_save_path)


if __name__ == "__main__":
    main()
