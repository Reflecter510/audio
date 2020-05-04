from __future__ import print_function
import os
import random
import numpy as np
#np.random.seed(1337)  # for reproducibility
import scipy
from keras.datasets import mnist
from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K, Input
from keras.utils.vis_utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from keras.backend import set_learning_phase
import matplotlib.pyplot as plt
'''语谱图  训练集  验证集 目录'''
train_dir = r'./data_set/specgram/train/'
validation_dir = r'./data_set/specgram/test/'
'''MFCC    训练集  验证集 目录'''
mfcc_train_dir = r'./data_set/mfcc/train/'
mfcc_validation_dir = r'./data_set/mfcc/test/'

model_dir="model/mix/save_model_mix.h5"
model = load_model(model_dir)

batch_size = 120           #每批多少张图
nb_classes = 2            #分多少类

'''语谱图数据集'''
input_shape=(256,256,3)
input_size=(256,256)
# 训练集数据增强
train_datagen = ImageDataGenerator(
     rescale=1./255,
    #width_shift_range=0.1,#0.2,
)
# 测试集
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    # 目标文件夹
    train_dir,
    # 规范化图片大小
    target_size=input_size,
    batch_size=batch_size,
    #shuffle=False,
    seed=2019,
    # 二分类
    #class_mode='binary'
    )
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=input_size,
    batch_size=batch_size,
    shuffle=False,
    #class_mode='binary'
    )
train_sum=len(train_generator.filenames)
vali_sum=len(validation_generator.filenames)

'''MFCC数据集'''
mfcc_input_shape=(256,256,3)
mfcc_input_size=(256,256)
# 训练集数据增强
mfcc_train_datagen = ImageDataGenerator(
     rescale=1./255,
    #width_shift_range=0.1,#0.2,
)
# 测试集
mfcc_test_datagen = ImageDataGenerator(rescale=1./255)
mfcc_train_generator = train_datagen.flow_from_directory(
    # 目标文件夹
    mfcc_train_dir,
    # 规范化图片大小
    target_size=input_size,
    batch_size=batch_size,
    #shuffle=False,
    seed=2019,
    # 二分类
    #class_mode='binary'
    )
mfcc_validation_generator = test_datagen.flow_from_directory(
    mfcc_validation_dir,
    target_size=input_size,
    batch_size=batch_size,
    shuffle=False,
    #class_mode='binary'
    )
mfcc_train_sum=len(mfcc_train_generator.filenames)
mfcc_vali_sum=len(mfcc_validation_generator.filenames)

'''构造生成器'''
def gen_flow_for_two_inputs(X1, X2):
    while True:
        X1i = X1.next()
        X2i = X2.next()
        if not (X1i[1]==X2i[1]).all():
            print("Match error!")
            continue
        yield [X1i[0], X2i[0]], X1i[1]

'''使用评估函数'''
eva=model.evaluate_generator(gen_flow_for_two_inputs(validation_generator,mfcc_validation_generator),steps=vali_sum/batch_size,verbose=1)
print(eva[1])