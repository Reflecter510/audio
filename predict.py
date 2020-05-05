from __future__ import print_function

import os
import shutil

from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
from feature_to_dataset import one_to_dataset
import numpy as np

class Prediction:
    model = ""
    input_shape = (256, 256, 3)
    input_size = (256, 256)
    batch_size = 5  # 每批多少张图

    def __init__(self,model_dir="model/mix/save_model_mix.h5",width=256,height=256):
        self.model = load_model(model_dir)
        self.input_shape = (width,height,3)
        self.input_size = (width,height)

    def predict(self,audio_dir = "data_audio/prediction"):

        clear_folder(os.path.join(audio_dir,"specgram/data"))
        clear_folder(os.path.join(audio_dir,"mfcc/data"))

        one_to_dataset(audio_dir,audio_dir,audio_dir)

        specgram_generator, mfcc_generator=self.gen_one_inputs(audio_dir)
        pred = self.model.predict_generator(gen_flow_for_two_inputs(specgram_generator,mfcc_generator),steps=len(mfcc_generator.filenames)/self.batch_size,verbose=1)
        #print(pred)
        predicted_class_indices = np.argmax(pred, axis=1)
        #print(predicted_class_indices)
        label = ["梅兰芳", "其他"]
        predictions = [label[i] for i in predicted_class_indices]
        # 建立预测结果和文件名之间的关系
        filenames = specgram_generator.filenames
        test_names = mfcc_generator.filenames
        if filenames != test_names:
            print("谱图无法对应！")
        sum = 0
        for idx in range(len(filenames)):
            sum += 1
            print('预测结果：%s , 概率为: %.2f' % (predictions[idx],pred[idx][predicted_class_indices[idx]]))
            if predictions[idx] == 'is' and predictions[idx] == filenames[idx][:2]:
                print(filenames[idx][:2])
            if predictions[idx] == 'not' and predictions[idx] == filenames[idx][:3]:
                print(filenames[idx][:3])
            print('文件名    %s' % filenames[idx])
            print('')
        print("总  数: " + str(sum))
        return "";

    def gen_one_inputs(self,audio_dir,specgram_dir = "data_audio/prediction/specgram",mfcc_dir = "data_audio/prediction/mfcc"):
        specgram_dir = audio_dir+"/specgram"
        mfcc_dir = audio_dir+"/mfcc"
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        mfcc_test_datagen = ImageDataGenerator(rescale=1. / 255)
        specgram_generator = test_datagen.flow_from_directory(
            specgram_dir,
            target_size=self.input_size,
            batch_size=self.batch_size,
            shuffle=False,
            # class_mode='binary'
        )
        mfcc_generator = mfcc_test_datagen.flow_from_directory(
            mfcc_dir,
            target_size=self.input_size,
            batch_size=self.batch_size,
            shuffle=False,
            # class_mode='binary'
        )
        return specgram_generator,mfcc_generator

'''删除目录下所有文件'''
def clear_folder(filepath):
    del_list = os.listdir(filepath)
    #print(del_list)
    #a = input("测试：")
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

'''构造生成器'''
def gen_flow_for_two_inputs(X1, X2):
    while True:
        X1i = X1.next()
        X2i = X2.next()
        if not (X1i[1] == X2i[1]).all():
            print("Match error!")
            continue
        yield [X1i[0], X2i[0]], X1i[1]

if __name__ == "__main__":
    p = Prediction()
    p.predict()
