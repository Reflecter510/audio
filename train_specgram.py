from __future__ import print_function
import os
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
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
img_rows, img_cols = 256, 256
# 卷积核的尺寸
input_shape=(img_rows,img_cols,3)
input_size=(img_rows,img_cols)
kernel_size = (3, 3)

# 使用的卷积滤波器的数量
nb_filters = 32
# 用于 max pooling 的池化面积
pool_size = (2, 2)

model_name="model/specgram/save_model.h5"
struct_name="model/specgram/save_struct.h5"
weights_name="model/specgram/save_weights.h5"

# *************** ImageDataGenerator *************** #
train_dir = r'./data_set/specgram/train/'
validation_dir = r'./data_set/specgram/test/'
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
plt.savefig("model/specgram/0_acc.png")
plt.cla()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model/specgram/0_loss.png")
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
plt.savefig("model/specgram/1_acc.png")
plt.cla()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model/specgram/1_loss.png")
plt.cla()

#model.fit(imgs,_label,batch_size=batch_size,epochs=nb_epoch2,verbose=1)
#plot_model(model, to_file='model-cnn.png')
model.save(model_name)
model.save_weights(weights_name)



