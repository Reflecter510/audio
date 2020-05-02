from __future__ import print_function
import os
import random
import numpy as np
# np.random.seed(510)  # for reproducibility
import scipy
from keras.applications import Xception, VGG16
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras import backend as K, Input
from PIL import Image
from keras.utils.vis_utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#######################################################################################################################
batch_size = 60  # 每批多少张图
nb_classes = 2  # 分多少类
nb_epoch = 5  # 迭代次数
width = 256
height = 256
input_shape = (256, 256, 3)
input_size = (256, 256)
model_name="model/specgram/save_model.h5"
struct_name="model/specgram/save_struct.h5"
weights_name="model/specgram/save_weights.h5"

# *************** ImageDataGenerator *************** #
train_dir = r'./data_set/specgram/train/'
validation_dir = r'./data_set/specgram/test/'
# 训练集数据增强
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=40,
    # width_shift_range=0.2,#0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
)
# 测试集
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    # 目标文件夹
    train_dir,
    # 规范化图片大小
    target_size=input_size,
    batch_size=batch_size,
    # shuffle=False,
    # 二分类
    # class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=input_size,
    batch_size=batch_size,
    shuffle=False,
    # class_mode='binary'
)
train_sum = len(train_generator.filenames)
vali_sum = len(validation_generator.filenames)

model = Sequential()
input=(Input(input_shape,name='input'))
x=Conv2D(64, (3, 3), padding="same", activation="relu",  name='conv1')(input)
x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1")(x)

x=Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2")(x)
x=MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name="pool2")(x)

x=Dropout(0.25)(x)

x=Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3")(x)
x=MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name="pool3")(x)

x=Dropout(0.25)(x)

x=Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4")(x)
x=MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name="pool4")(x)

x=Dropout(0.25)(x)

x=Flatten()(x)
x=Dense(128, activation="relu", name="fc1")(x)
x=Dropout(0.5)(x)
output=Dense(nb_classes, activation="softmax", name="fc2")(x)

model=Model(inputs=input,outputs=output,name="net")
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["acc"])

file = open(struct_name,"w")
file.write(model.to_json())
file.close()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_sum / batch_size,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=vali_sum / batch_size,
)
'''画图'''
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model/specgram/specgram_acc.png")
plt.cla()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model/specgram/specgram_loss.png")
plt.cla()

model.save(model_name)
model.save_weights(weights_name)
