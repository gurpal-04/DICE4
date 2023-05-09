from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import os
import pickle

os.sys.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

img = image.load_img("Test/HR-S-027.jpg")
# plt.imshow(img)
# cv2.imread(
#   "E:\AI\ML\Dataset\Segmented Medicinal Leaf Images\Alpinia Galanga (Rasna)\AG-S-001.jpg"
# )

train = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory('Train',
                                          target_size=(200, 200),
                                          batch_size=3,
                                          class_mode='binary')
validation_dataset = validation.flow_from_directory('Validation',
                                                    target_size=(200, 200),
                                                    batch_size=3,
                                                    class_mode='binary')

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer=RMSprop(learning_rate=0.001),
              metrics=["accuracy"])

model_fit = model.fit(train_dataset,
                      steps_per_epoch=2,
                      epochs=15,
                      validation_data=validation_dataset)

# path = 'Test'
# for i in os.listdir(path):
#   img = image.load_img(path + "//" + i, target_size=(200, 200))
#   plt.imshow(img)
#   plt.show()

#   X = image.img_to_array(img)
#   X = np.expand_dims(X, axis=0)
#   images = np.vstack([X])
#   val = model.predict(images)

#   if val == 0:
#     print('Hibiscus')
#   elif val == 1:
#     print("Mint")

pickle.dump(model, open('plant_detection_model.pkl', 'wb'))
