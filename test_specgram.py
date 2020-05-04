from __future__ import print_function
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from keras.backend import set_learning_phase
#set_learning_phase(1)

model_dir=('model/specgram/save_model.h5')  #预测模型
test_dir = r'./data_set/specgram/test/'              #预测文件夹

model = load_model(model_dir)

batch_size = 1#每批多少张图
nb_classes = 2            #分多少类
input_size=(256,256)
train_dir = r'./data_set/specgram/train/'
train_datagen = ImageDataGenerator(
    rescale=1./255,
   # rotation_range=40,
   width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
)
train_generator = train_datagen.flow_from_directory(
    # 目标文件夹
    train_dir,
    # 规范化图片大小
    target_size=input_size,
    batch_size=batch_size,
    shuffle=False,
    # 二分类
    #class_mode='binary'
    )
# 测试集
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  #width_shift_range=0.2,
                                  )
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_size,
    batch_size=batch_size,
    shuffle=False,
    #class_mode='binary'
)
train_sum=len(train_generator.filenames)
test_sum=len(test_generator.filenames)


'''对生成器预测'''
test_generator.reset()
pred = model.predict_generator(test_generator,steps=test_sum/batch_size,verbose=1)
print(pred)
predicted_class_indices = np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
label = dict((v,k) for k,v in labels.items())
# 建立代码标签与真实标签的关系
predictions = [label[i] for i in predicted_class_indices]
#建立预测结果和文件名之间的关系
filenames = test_generator.filenames
cnt=0
sum=0
for idx in range(len(filenames )):
    sum+=1
    print('预测  %s' % (predictions[idx]))
    '''
    if predictions[idx] =='not':
        cnt+=1
    '''
    if predictions[idx] == 'is' and predictions[idx]==filenames[idx][:2]:
        print(filenames[idx][:2])
        cnt+=1
    if predictions[idx] == 'not' and predictions[idx]==filenames[idx][:3]:
        print(filenames[idx][:3])
        cnt+=1
    print('文件名    %s' % filenames[idx])
    print('')
print(predicted_class_indices)
print("正确数"+str(cnt))
print("总  数"+str(sum))
print(cnt/sum)


'''使用评估函数'''
test_generator.reset()
eva=model.evaluate_generator(test_generator,steps=test_sum/batch_size,verbose=0)
print(eva[1])


'''逐个读取图片来预测
# 加载图片
#img_names = [r'C:\ Users\ admin\PycharmProjects\ audio\语谱图CNN\ train\is\】穆桂英挂帅 (Live)-57.png']
img_names=test_generator.filenames
print(img_names)
# np.transpose:对数组进行转置，返回原数组的视图
# scipy.misc.imresize：重新调整图片的形状
# scipy.misc.imread：将图片读取出来，返回np.array类型
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(test_dir+img_name), (80, 256)), (1, 0, 2)).astype('float32')
		for img_name in img_names]
imgs = np.array(imgs) / 255.0		# 归一化
imgs=imgs[... ,0:3]
predict = model.predict(imgs,batch_size=batch_size)
predict=np.argmax(predict,axis=1)
cnt=0
sum=0
for i in range(0,len(predict)):
    sum+=1
    if label[predict[i]]=='is' and label[predict[i]]==img_names[i][:2]:
        cnt+=1
    if label[predict[i]]=='not' and label[predict[i]]==img_names[i][:3]:
        cnt+=1
    print(img_names[i]+" :"+label[predict[i]])
print("正确数"+str(cnt))
print("总  数"+str(sum))
print(cnt/sum)
'''