from __future__ import print_function
import os
import random
import numpy as np
#np.random.seed(510)  # for reproducibility
import scipy
from keras.applications import Xception, VGG16
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from PIL import Image
from keras.utils.vis_utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#######################################################################################################################
batch_size = 60           #每批多少张图
nb_classes = 2            #分多少类
nb_epoch1 = 2              #迭代次数
nb_epoch2 = 1#00
# 输入数据的维度
img_rows, img_cols = 256, 80
# 使用的卷积滤波器的数量
nb_filters = 32
# 用于 max pooling 的池化面积
pool_size = (2, 2)
# 卷积核的尺寸
kernel_size = (3, 3)
input_shape=(80,80,3)
input_size=(80,80)
model_name="save_model.h5"
struct_name="save_struct.h5"
weights_name="save_weights.h5"

# *************** ImageDataGenerator *************** #
train_dir = r'./data_set/train/'
validation_dir = r'./data_set/test/'
# 训练集数据增强
train_datagen = ImageDataGenerator(
     rescale=1./255,
    #rotation_range=40,
    #width_shift_range=0.2,#0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
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


'''
# 加载图片
img_names=train_generator.filenames
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(train_dir+img_name), (80, 256)), (1, 0, 2)).astype('float32')
		for img_name in img_names]
imgs = np.array(imgs) / 255.0		# 归一化
imgs=imgs[... ,0:3]
_label=[]
for i in range(0,int(train_sum/2)):
    _label.append([1,0])
for i in range(int(train_sum/2),int(train_sum)):
    _label.append([0,1])
_label=np.array(_label)
'''

# 构建不带分类器的预训练模型
base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=input_shape)

'''得到特征向量
print(validation_generator.samples)
feature=base_model.predict_generator(validation_generator)
print(np.shape(feature))
feature=base_model.predict_generator(train_generator)
print(np.shape(feature))
'''


# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(1024, activation='relu')(x)

# 添加一个分类器，假设我们有2个类
predictions = Dense(nb_classes, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])

file = open(struct_name,"w")
file.write(model.to_json())
file.close()

# 在新的数据集上训练几代
# *************** Run *************** #
''''''
history = model.fit_generator(
      train_generator,
      steps_per_epoch=  train_sum/batch_size,
      epochs=nb_epoch1,
      validation_data=validation_generator,
      validation_steps= vali_sum/batch_size
)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("0_acc.png")
plt.cla()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("0_loss.png")
plt.cla()
#model.fit(imgs,_label,batch_size=batch_size,epochs=nb_epoch1,verbose=1)


#model.save("save_model_0.h5")
# 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
# 我们会锁住底下的几层，然后训练其余的顶层。

# 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# 我们选择训练最上面的两个 Inception block
# 也就是说锁住前面249层，然后放开之后的层。
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# 我们需要重新编译模型，才能使上面的修改生效   sparse_categorical_crossentropy
# 让我们设置一个很低的学习率，使用 SGD 来微调
from keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['acc'])

# 我们继续训练模型，这次我们训练最后两个 Inception block
# 和两个全连接层
''''''
history = model.fit_generator(
      train_generator,
      steps_per_epoch=   train_sum/batch_size,
      epochs=nb_epoch2,
      validation_data=validation_generator,
      validation_steps=   vali_sum/batch_size,
      )
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("1_acc.png")
plt.cla()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("1_loss.png")
plt.cla()

#model.fit(imgs,_label,batch_size=batch_size,epochs=nb_epoch2,verbose=1)
#plot_model(model, to_file='model-cnn.png')
model.save(model_name)
model.save_weights(weights_name)



'''测试
train_generator.reset()
pred = model.predict_generator(train_generator,steps=train_sum/batch_size,verbose=1)
print(pred)
predicted_class_indices = np.argmax(pred,axis=1)
print(predicted_class_indices)
labels = (train_generator.class_indices)
label = dict((v,k) for k,v in labels.items())
# 建立代码标签与真实标签的关系
predictions = [label[i] for i in predicted_class_indices]
#建立预测结果和文件名之间的关系
filenames = train_generator.filenames
cnt=0
sum=0
for idx in range(len(filenames )):
    sum+=1
    print('预测  %s' % (predictions[idx]))
    if predictions[idx] == 'is' and predictions[idx]==filenames[idx][:2]:
        print(filenames[idx][:2])
        cnt+=1
    if predictions[idx] == 'not' and predictions[idx]==filenames[idx][:3]:
        print(filenames[idx][:3])
        cnt+=1
    print('文件名    %s' % filenames[idx])
    print('')
print("正确数"+str(cnt))
print("总  数"+str(sum))
print(cnt/sum)
'''
'''评估
from keras.backend import set_learning_phase
set_learning_phase(1)
print("learning phase=1")
train_generator.reset()
score = model.evaluate_generator(train_generator,steps=train_sum/batch_size,verbose=0)
print('Train score:', score[0])
print('Train accuracy:', score[1])
set_learning_phase(0)
print("learning phase=0")
train_generator.reset()
score = model.evaluate_generator(train_generator,steps=train_sum/batch_size,verbose=0)
print('Train score:', score[0])
print('Train accuracy:', score[1])
'''



'''
train_path1 = './pop_train_is/pic_one/'  # 是  的训练路径
train_path2 = './pop_train_not/pic_one/'  # 不是  的训练路径
test_path1 = './pop_test_is/pic_one/'  # 是  的测试路径
test_path2 = './pop_test_not/pic_one/'  # 不是  的测试路径

def get_one_(path,file,type):
    image=Image.open(path+file)
    print(np.shape(image))
    if np.shape(image)[0]!=256 or np.shape(image)[1]!=80:
        return [],[]
    image = np.reshape(image, [256,80,4])
    label=[]
    if type==1:
        label=[0,1]
    else:
        label=[1,0]
    return image,label

def generate_data(train_path1,train_path2,sum=-1):
    X=[]
    Y=[]
    train1_list = os.listdir(train_path1)  # 文件名列表
    train2_list = os.listdir(train_path2)
    cnt_train1 = len(train1_list)  # 训练数据1的长度
    cnt_train2 = len(train2_list)  # 训练数据2的长度
    if sum ==-1:
        sum=cnt_train1+cnt_train2
    count = 0
    train_acc = 0.0
    # 随机选取数据训练
    while (train1_list or train2_list) and count<sum:
        get = random.randint(0, 1)  # 获得0和1的随机数
        #print(get)
        if get == 0 and train1_list:
            b_image, b_label = get_one_(train_path1, train1_list[0], 1)  # 获取数据
            if len(b_image) == 0 and len(b_label) == 0:
                train1_list.pop(0)
                count+=1
                continue
            X.append(b_image)
            Y.append((b_label))
            print(train1_list[0])  # 输出文件名
            train1_list.pop(0)
            count += 1
        elif get == 1 and train2_list:
            b_image, b_label = get_one_(train_path2, train2_list[0], 0)  # 获取数据
            if len(b_image) == 0 and len(b_label) == 0:
                train2_list.pop(0)
                count+=1
                continue
            X.append(b_image)
            Y.append((b_label))
            print(train2_list[0])  # 输出文件名
            train2_list.pop(0)
            count += 1
    X=np.array(X)
    Y=np.array(Y)
    return X,Y
'''

'''
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
'''
'''
X_train,Y_train=generate_data(train_path1,train_path2)
X_test,Y_test=generate_data(test_path1,test_path2)
print("数据",type(X_train),np.shape(X_train))
print("标签",type(Y_train),np.shape(Y_train))
print("test数据",type(X_test),np.shape(X_test))
print("test标签",type(Y_test),np.shape(Y_test))
'''

'''简单的网络
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

#plot(model, to_file='model-cnn.png')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
'''


'''
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
'''
