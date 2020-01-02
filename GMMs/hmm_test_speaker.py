# -*- coding: utf-8 -*-
#test_gender.py
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
from Proprecess import Preprocess
import warnings
import librosa
warnings.filterwarnings("ignore")
import time
from matplotlib import pyplot as plt


class hmm_test():

    def __init__(self,modelspath,file_path):
        self.hmm_files=[os.path.join(modelspath,fname) for fname in os.listdir(modelspath) if fname.endswith('.hmm')]
        self.models=[pickle.load(open(fname,'rb')) for fname in self.hmm_files]
        self.speakers   = [fname.split("\\")[-1].split(".hmm")[0] for fname in self.hmm_files]
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
                self.hmm    = self.models[i]         #checking with each model one by one
                scores = np.array(self.hmm.score(vector))
                log_likelihood[i] = scores.sum()
                print(log_likelihood[i])

            winner = np.argmax(log_likelihood)
            if outScroe==True:
                print("最高分:",log_likelihood[winner])

            print ("\t预测为 - ", self.speakers[winner])

            if outPredict==True:
                print(self.hmm.predict(vector))

            if self.speakers[winner]=="music2_train":
                ac+=1
            time.sleep(1.0)
        print("准确率：", ac / all)

    def get_distribution(self):
        pass

    def test_one_model(self,model_name):
        self.hmm=self.models[self.speakers.index(model_name)]
        all = 0.0
        ac = 0.0
       # print(self.hmm.get_params())
        print("输出根据数据训练出来的π")
        print(self.hmm.startprob_)
        print("输出根据数据训练出来的A")
        print(self.hmm.transmat_)
        print("输出根据数据训练出来的B")
        # print(_hmm.emissionprob_)

        for file in os.listdir(self.file_paths):  # 读取数据集文件夹中的每一个文件
            file_path = os.path.join(self.file_paths, file)
            print(file_path)
            audio, sr = librosa.load(file_path)
            vector = extract_features(audio, sr)
            all += 1
            scores=np.array(self.hmm.score(vector))
            print("分数：",scores.sum())
            ''' '''
            #z=self.hmm.score_samples(vector)
            #print(z)

            Z = self.hmm.predict(vector)
            xValue = list(range(0, len(Z)))
            print(np.shape(Z))
            # plt.scatter(xValue,Z, s=20, c="#ff1212", marker='o')

            plt.hist(Z, bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
            plt.title("Z")
            plt.show()


if __name__=="__main__":
    modelspath = "hmm_speaker_models\\"
    # modelpath = "model\\"
                                               # 测试数据集的文件夹
    file_paths = "music1_test/"

    Hmm=hmm_test(modelspath=modelspath,file_path=file_paths)
    Hmm.test_oneByOne(outScroe=True)
    #Hmm.test_one_model("music2_train")







'''
#模型文件夹
modelpath = "hmm_speaker_models\\"
#modelpath = "model\\"


hmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.hmm')]
#加载模型
models    = [pickle.load(open(fname,'rb')) for fname in hmm_files]
speakers   = [fname.split("\\")[-1].split(".hmm")[0] for fname in hmm_files]

# 测试数据集的文件夹
file_paths = "music1_test/"

all = 0.0
ac = 0.0
_hmm = models[0]
print("输出根据数据训练出来的π")
print(_hmm.startprob_)
print("输出根据数据训练出来的A")
print(_hmm.transmat_)
print("输出根据数据训练出来的B")
# print(_hmm.emissionprob_)
for file in os.listdir(file_paths):         #读取数据集文件夹中的每一个文件
    file_path = os.path.join(file_paths, file)
    print(file_path)
    audio,sr=librosa.load(file_path)
    #vector = Preprocess.enframe(audio, sr, 256)
    vector   = extract_features(audio,sr)
    all += 1
    
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        _hmm    = models[i]         #checking with each model one by one
        scores = np.array(_hmm.score(vector))
        log_likelihood[i] = scores.sum()
        #print(log_likelihood[i])
    winner = np.argmax(log_likelihood)
    print ("\t预测为 - ", speakers[winner])
    if speakers[winner]=="music2_train":
        ac+=1
    

    scores=np.array(_hmm.score(vector))
    score=scores.sum()          #分数是  概率取对数ln
    print(score)
    #print(hmm.predict(vector))

   
    Z=_hmm.predict(vector)
    xValue = list(range(0, len(Z)))
    print(np.shape(Z))
    #plt.scatter(xValue,Z, s=20, c="#ff1212", marker='o')

    plt.hist(Z ,bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.title("Z")
    plt.show()
    time.sleep(1.0)
print("准确率：",ac/all)

'''