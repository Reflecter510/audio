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
#set_learning_phase(1)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''语谱图  模型文件夹'''
model_dir=('model/specgram/save_struct.h5')
weights_dir=("model/specgram/save_weights.h5")
'''MFCC    模型文件夹'''
mfcc_model_dir=('model/mfcc/save_struct.h5')
mfcc_weights_dir=("model/mfcc/save_weights.h5")
'''语谱图  训练集  验证集 目录'''
train_dir = r'./data_set/specgram/train/'
validation_dir = r'./data_set/specgram/test/'
'''MFCC    训练集  验证集 目录'''
mfcc_train_dir = r'./data_set/mfcc/train/'
mfcc_validation_dir = r'./data_set/mfcc/test/'

file = open(model_dir,"r")
struct=file.read()
file = open(mfcc_model_dir,"r")
mfcc_struct=file.read()

'''融合模型'''
model_1 = model_from_json(struct)
model_1.load_weights(weights_dir)
# 我们选择训练最上面的两个 Inception block
# 也就是说锁住前面249层，然后放开之后的层。
for layer in model_1.layers:#[:249]:
   layer.trainable = False
#for layer in model_1.layers[249:]:
#   layer.trainable = True
model_2 = model_from_json(mfcc_struct)
model_2.load_weights(weights_dir)
# 我们选择训练最上面的两个 Inception block
# 也就是说锁住前面249层，然后放开之后的层。
for layer in model_2.layers:#[:249]:
   layer.trainable = False
#for layer in model_2.layers[249:]:
#   layer.trainable = True

out1=model_1.get_layer("mixed10").output
out2=model_2.get_layer("mixed10").output
for i, layer in enumerate(model_1.layers):
   layer.name=layer.name+"_Zero"#更改名字，以防冲突
   #print(i, layer.name)
in1 = model_1.layers[0].output
in2 = model_2.layers[0].output
in_0=Input(name="MixInput",shape=(12,1,2048),dtype=float)
x = concatenate([out1, out2],name="Mix_Concat")
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output_ = Dense(2, activation='softmax', name='output')(x)
model = Model(inputs=[in1,  in2], outputs=[output_])
model.summary()

'''锁层操作
# 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
for i, layer in enumerate(model.layers):
   print(i, layer.name)
lock=input("锁住多少层：")
# 我们选择训练最上面的两个 Inception block
# 也就是说锁住前面249层，然后放开之后的层。
for layer in model.layers[:lock]:
   layer.trainable = False
for layer in model.layers[lock:]:
   layer.trainable = True
'''
'''编译模型'''
from keras.optimizers import SGD,rmsprop
model.compile(optimizer=rmsprop(lr=0.0001), loss='categorical_crossentropy',metrics=['acc'])

'''训练设置'''
nb_epoch=2
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
    target_size=mfcc_input_size,
    batch_size=batch_size,
    #shuffle=False,
    seed=2019,
    # 二分类
    #class_mode='binary'
    )
mfcc_validation_generator = test_datagen.flow_from_directory(
    mfcc_validation_dir,
    target_size=mfcc_input_size,
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

'''训练'''
history = model.fit_generator(
    gen_flow_for_two_inputs(train_generator,mfcc_train_generator),
    steps_per_epoch=   train_sum/batch_size,
    epochs=nb_epoch,
    validation_data=gen_flow_for_two_inputs(validation_generator,mfcc_validation_generator),
    validation_steps=   vali_sum/batch_size,
    verbose=1
    )
'''画图'''
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model/mix/mix_acc.png")
plt.cla()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model/mix/mix_loss.png")
plt.cla()
'''保存模型'''
model.save("model/mix/save_model_mix.h5")
model.save_weights("model/mix/save_weights_mix.h5")
