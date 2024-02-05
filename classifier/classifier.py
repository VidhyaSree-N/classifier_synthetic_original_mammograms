import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

class classifier:
    def load_data(train_image_folder, train_npz_folder, test_size=0.3, random_state=42):
        image_files = [os.path.join(train_image_folder, f) for f in os.listdir(train_image_folder) if f.endswith('.png')]
        npz_files = [os.path.join(train_npz_folder, f) for f in os.listdir(train_npz_folder) if f.endswith('.npz')]

        train_files, test_files = train_test_split(list(zip(image_files, npz_files)), test_size=test_size, random_state=random_state)
        val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=random_state)

        return train_files, val_files, test_files

    def custom_data_generator(file_list, batch_size, img_height, img_width):
        total_files = len(file_list)
        indices = np.arange(total_files)
        np.random.shuffle(indices)

        while True:
            for i in range(0, total_files, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_images = []
                batch_labels = []

                for idx in batch_indices:
                    img_file, npz_file = file_list[idx]

                    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (img_width, img_height))
                    image = np.expand_dims(image, axis=-1)
                    image = image / 255.0
                    batch_images.append(image)
                    batch_labels.append(1)

                    with np.load(npz_file, allow_pickle=True) as data:
                        npz = data['data']
                    npz = np.expand_dims(npz, axis=-1)

                    batch_images.append(npz)
                    batch_labels.append(0)

                batch_images = np.array(batch_images)
                batch_labels = to_categorical(batch_labels, num_classes=2)

                indices_within_batch = np.arange(len(batch_labels))
                np.random.shuffle(indices_within_batch)

                yield batch_images[indices_within_batch], batch_labels[indices_within_batch]

    def build_model(img_height, img_width, NUMBER_OF_CLASSES):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUMBER_OF_CLASSES, activation='sigmoid'))

        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer="adadelta",
                      metrics=['accuracy'])

        return model

    def train_model(model, train_generator, val_generator, batch_size, early_stopping):
        try:
            history = model.fit(
                train_generator,
                steps_per_epoch=300,
                epochs=1,
                validation_data=val_generator,
                validation_steps=10,
                verbose=1,
                callbacks=[early_stopping]
            )
        except Exception as e:
            print("An error occurred during training:", str(e))

    def evaluate_model(model, test_generator, batch_size):
        test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_files) // batch_size)
        print(f"Test Accuracy: {test_accuracy}")

    def display_predictions(self, test_generator):
            for i in range(5):
                test_sample, test_labels = next(test_generator)
                predictions = self.model.predict(test_sample)

                # Display the predicted label
                plt.subplot(1, 2, 2)
                predicted_label = np.argmax(predictions[i])
                actual_label = np.argmax(test_labels[i])
                plt.title(f"Predicted Label: {predicted_label}, Actual Label: {actual_label}")
                plt.imshow(test_sample[i], cmap='gray')
                plt.axis('off')

                plt.tight_layout()
                plt.show()

            # 1 for Synthetic and 0 for original
            predictions = self.model.predict(test_generator, steps=len(self.test_files) // self.batch_size, verbose=1)
            predicted_labels = np.argmax(predictions, axis=1)
            return predicted_labels
