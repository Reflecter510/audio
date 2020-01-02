# -*- coding: utf-8 -*-
#test_gender.py
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from GMMs.speakerfeatures import extract_features
import warnings
import librosa
warnings.filterwarnings("ignore")
import time


class gmm_test():

    def __init__(self,modelspath,file_path):
        self.gmm_files=[os.path.join(modelspath,fname) for fname in os.listdir(modelspath) if fname.endswith('.gmm')]
        self.models=[pickle.load(open(fname,'rb')) for fname in self.gmm_files]
        self.speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname in self.gmm_files]
        self.features = np.asarray(())
        self.file_paths=file_path

    def load_all_test_data(self):
        for file in os.listdir(self.file_paths):  # 读取数据集文件夹中的每一个文件
            file_path = os.path.join(self.file_paths, file)
            print(file_path)
            audio, sr = librosa.load(file_path)
            vector = extract_features(audio, sr)
            if self.features.size == 0:
                self.features = vector
            else:
                self.features = np.vstack((self.features, vector))

    def load_one_test_data(self):
        pass

    def test_oneByOne(self,outPredict=False,outScroe=False):
        all = 0.0
        ac = 0.0
        for file in os.listdir(self.file_paths):  # 读取数据集文件夹中的每一个文件
            file_path = os.path.join(self.file_paths, file)
            print(file_path)
            audio, sr = librosa.load(file_path)
            vector = extract_features(audio, sr)
            all += 1

            log_likelihood = np.zeros(len(self.models))
            for i in range(len(self.models)):
                self.gmm    = self.models[i]         #checking with each model one by one
                scores = np.array(self.gmm.score(vector))
                log_likelihood[i] = scores.sum()
                #print(log_likelihood[i])

            winner = np.argmax(log_likelihood)
            if outScroe==True:
                print("分数:",log_likelihood[winner])

            print ("\t预测为 - ", self.speakers[winner])

            if outPredict==True:
                print(self.gmm.predict(vector))

            if self.speakers[winner]=="music2_train":
                ac+=1
            time.sleep(1.0)
        print("准确率：", ac / all)

    def get_distribution(self):
        pass

    def test_one_model(self,model_name):
        self.gmm=self.models[self.speakers.index(model_name)]
        all = 0.0
        ac = 0.0
        print(self.gmm.get_params())
        for file in os.listdir(self.file_paths):  # 读取数据集文件夹中的每一个文件
            file_path = os.path.join(self.file_paths, file)
            print(file_path)
            audio, sr = librosa.load(file_path)
            vector = extract_features(audio, sr)
            all += 1
            scores=np.array(self.gmm.score(vector))
            print("分数:",scores.sum())
            z=self.gmm.score_samples(vector)
            sum=0.0000
            for each in z:
                sum+=np.exp(each)
            print(sum)



if __name__=="__main__":
    modelspath = "speaker_models\\"
    #modelpath = "model\\"
    file_paths = "music1_test/"

    Gmm=gmm_test(modelspath=modelspath,file_path=file_paths)
    Gmm.test_oneByOne(outScroe=True)
    #Gmm.test_one_model("music2_train")



