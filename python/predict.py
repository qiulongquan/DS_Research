import numpy as np
import keras
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from keras.applications import imagenet_utils, vgg16

print(tf.__version__)
# parameters setting
classes = ['zero', 'one', 'two', 'three', 'four',
           'five', 'six', 'seven', 'eight', 'nine']
model_path = 'python/sign_language_vgg16_1.h5'
# setting test image path
img_path = "python/hand_sign_digit_data/test/6/6_166.jpg"
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img)

print(img_array.shape)
# expected conv2d_input to have 4 dimensions, but got array with shape (100, 100, 3)
# 需要变成(1, 100, 100, 3)这个形式的矩阵，一张图片所以开头样品数量是1
pImg = np.expand_dims(img_array, axis=0)/255
print(pImg.shape)

# load model
sign_language_vgg16 = tf.keras.models.load_model(model_path)
sign_language_vgg16.summary()
# [0]表示结果中选取第一个数组（10个结果数字）然后排序输出
prediction = sign_language_vgg16.predict(pImg)[0]
print("qiulongquan_prediction={}".format(prediction))
print(np.argmax(prediction))

# [::-1]降序排序  [-5:]取前5个值  argsort表示返回的不是值，而是index
top_indices = prediction.argsort()[-5:][::-1]
result = [(classes[i], prediction[i]) for i in top_indices]
for x in result:
    print(x)
