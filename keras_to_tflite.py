import tensorflow as tf
'''
将keras训练保存的模型转换为tflite模型

训练的模型需要 在出错的源码前 改  lr 成 leraning_rate
也就是lr = kwargs.pop('learning_rate', lr)
'''

model_category = "specgram"

model_path = "model/"+model_category +"/save_model.h5"
save_path = "model/"+model_category+"/save_model.tflite"

converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
tflite_model = converter.convert()
open(save_path, "wb").write(tflite_model)
