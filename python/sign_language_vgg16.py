import keras
import os
import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflowjs as tfjs

base_path = "hand_sign_digit_data"
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
    # 32 is kernel count    (3,3) is kernel dimension
    32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# MaxPooling2D (2,2) is  pooling dimension
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Flatten())

# 这个是在最后部分使用dropout 部分note失活，也可以在每一层都使用dropout。dropout的目的是防止过拟合
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

classes = ['zero', 'one', 'two', 'three', 'four',
           'five', 'seven', 'eight', 'nine']

train_dir = os.path.join(base_path, 'train')
validation_dir = os.path.join(base_path, 'validation')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=64,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=64,
    class_mode='categorical')

history = model.fit_generator(
    # 这里需要计算一下图片的数量和每一个batch的数量*一个epoch的step数量是不是匹配
    train_generator,
    steps_per_epoch=51,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=14)

model.save('sign_language_vgg16_new.h5')

# convert the vgg16 model into tf.js model
print(keras.__version__)
print(tfjs.__version__)
save_path = '../nodejs/static/sign_language_vgg16_new'
tfjs.converters.save_keras_model(model, save_path)
print("[INFO] saved tf.js vgg16 model to disk..")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
