#train_models.py
import os
import pickle
from scipy import signal
import wave

import librosa
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from GMMs.speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")


class gmm_train():
    def __init__(self,dest,file_paths,sample_rate=16000,offset=0):
        self.dest=dest
        self.file_paths=file_paths
        self.sample_rate=sample_rate
        self.offset=offset
        self.features= np.asarray(())

    def init_model(self,n_components=200, max_iter=20000, covariance_type='diag', n_init=3, verbose=3,verbose_interval=500):
        self.gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type=covariance_type, n_init=n_init, verbose=verbose,verbose_interval=verbose_interval)

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

    def train(self):
        self.gmm.fit(self.features)
        self.save_model()

    def save_model(self):
        picklefile = file_paths[0:-1] + ".gmm"       # dumping the trained gaussian model
        pickle.dump(self.gmm, open(dest + picklefile, 'wb'))
        print('+ modeling completed for speaker:', picklefile, " with data point = ", self.features.shape)

if __name__=="__main__":
    file_paths = "music2_train/"          #数据集文件目录，一次一个模型
    dest = "speaker_models/"        #训练生成的模型目录
    Gmm=gmm_train(dest=dest,file_paths=file_paths)
    Gmm.init_model(n_components=60)
    Gmm.load_all_test_data()
    Gmm.train()












    '''
    #参数
    #file_paths="abjones/"
    file_paths = "music2_train/"          #数据集文件目录，一次一个模型
    dest = "speaker_models/"        #训练生成的模型目录
    sample_rate=16000               #采样率
    offset=0                        #偏移量
    duration=None

    features = np.asarray(())
    for file in os.listdir(file_paths):         #读取数据集文件夹中的每一个文件
        file_path = os.path.join(file_paths, file)
        print(file_path)
        # 读取语音，并获取....mfcc系数作为特征...................................................
        #y, sr_orig = librosa.load(path=file_path, sr=sample_rate,offset=offset, duration=duration)
        #vector = librosa.feature.mfcc(y, sr=sr_orig,n_mfcc=40)
        #sr, audio = read(file_path)
        audio, sr = librosa.load(file_path,sr=sample_rate)
        vector = extract_features(audio, sr)
# extract 40 dimensional MFCC & delta MFCC features
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    gmm = GaussianMixture(n_components=200, max_iter=20000, covariance_type='diag', n_init=3,verbose=3,verbose_interval=500)
    gmm.fit(features)
    print("权重:",gmm.weights_)
    #print("::",gmm._get_parameters())
    # dumping the trained gaussian model
    picklefile = file_paths[0:-1]+ ".gmm"
    pickle.dump(gmm, open(dest + picklefile, 'wb'))
    print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
    features = np.asarray(())
    '''


    '''
#path to training data
source   = "development_set/"


#path where training speakers will be saved
dest = "speaker_models/"
train_file = "development_set_enroll.txt"
file_paths = open(train_file,'r')
count = 1
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print (path)
    # read the audio
    sr,audio = read(source + path)
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    if count == 5:    
        gmm = GaussianMixture(n_components = 16, max_iter = 2000, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        pickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
'''