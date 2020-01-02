#train_models.py
import os
import pickle
from scipy import signal
from Proprecess import Preprocess
import librosa
import numpy as np
np.set_printoptions(threshold = 1e6)     #  threshold表示输出数组的元素数目
from speakerfeatures import extract_features
import warnings
from hmmlearn import hmm
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

class hmm_train():
    def __init__(self,dest,file_paths,sample_rate=16000,offset=0):
        self.dest=dest
        self.file_paths=file_paths
        self.sample_rate=sample_rate
        self.offset=offset
        self.features= np.asarray(())

    def init_model(self,n_components=5,n_iter=10000,tol=0.001,verbose=True):
        self.hmm = hmm.GMMHMM(n_components=n_components,n_iter=n_iter,tol=tol,verbose=verbose)

    def load_all_test_data(self):
        for file in os.listdir(self.file_paths):  # 读取数据集文件夹中的每一个文件
            file_path = os.path.join(self.file_paths, file)
            print(file_path)
            audio, sr = librosa.load(file_path)
            self.vector = extract_features(audio, sr)
            #print(np.shape(self.vector))
            if self.features.size == 0:
                self.features = self.vector
            else:
                self.features = np.vstack((self.features, self.vector))

    def train(self):
        self.hmm.fit(self.features)
        self.save_model()

        print("输出最后一段音频预测的隐藏状态")
        Z = self.hmm.predict(self.vector)
        xValue = list(range(0, len(Z)))
        print(np.shape(Z))
        plt.scatter(xValue, Z, s=20, c="#ff1212", marker='o')
        plt.title("Z")
        plt.show()
        print("输出根据数据训练出来的π")
        print(self.hmm.startprob_)
        print("输出根据数据训练出来的A")
        print(self.hmm.transmat_)
        print("输出根据数据训练出来的B")
        # print(_hmm.emissionprob_)

    def save_model(self):
        picklefile = file_paths[0:-1] + ".hmm"       # dumping the trained gaussian model
        pickle.dump(self.hmm, open(dest + picklefile, 'wb'))
        print('+ modeling completed for speaker:', picklefile, " with data point = ", self.features.shape)



if __name__=="__main__":

    file_paths = "music2_train/"          #数据集文件目录，一次一个模型
    dest = "hmm_speaker_models/"        #训练生成的模型目录

    Hmm=hmm_train(dest=dest,file_paths=file_paths)
    Hmm.init_model(n_components=5)
    Hmm.load_all_test_data()
    #Hmm.train()




















    '''
    sample_rate=16000               #采样率
    offset=0                        #偏移量
    duration=None
    features = np.asarray(())

    for file in os.listdir(file_paths):         #读取数据集文件夹中的每一个文件
        file_path = os.path.join(file_paths, file)
        print(file_path)
                                                            # 读取语音，并提取mfcc系数作为特征
        audio, sr = librosa.load(file_path,sr=sample_rate)
        #vector=Preprocess.enframe(audio,256,128)
        vector = extract_features(audio, sr)      # 提取 40 维的 MFCC & delta MFCC 特征
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    _hmm=hmm.MultinomialHMM(n_components=10,n_iter=10000,tol=0.001,verbose=True)       #定义模型

    _hmm.fit(features)    #喂数据训练

    # 保存训练模型
    picklefile = file_paths[0:-1]+ ".hmm"
    pickle.dump(_hmm, open(dest + picklefile, 'wb'))
    print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)

    print("输出最后一段音频预测的隐藏状态")
    Z=_hmm.predict(vector)
    xValue = list(range(0, len(Z)))
    print(np.shape(Z))
    plt.scatter(xValue,Z, s=20, c="#ff1212", marker='o')
    plt.title("隐藏状态Z")
    plt.show()
    print("输出根据数据训练出来的π")
    print(_hmm.startprob_)
    print("输出根据数据训练出来的A")
    print(_hmm.transmat_)
    print("输出根据数据训练出来的B")
    #print(_hmm.emissionprob_)
    features = np.asarray(())
    '''