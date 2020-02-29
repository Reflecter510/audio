import tensorflow as tf
'''
训练的模型需要 在出错的源码前 改  lr 成 leraning_rate
也就是lr = kwargs.pop('learning_rate', lr)
'''
converter = tf.lite.TFLiteConverter.from_keras_model_file(r'C:\Users\admin\PycharmProjects\audio\语谱图CNN\save_model.h5')
tflite_model = converter.convert()
open("save_model.tflite", "wb").write(tflite_model)
