import wave

import numpy as np
import sklearn

from GMMs import zcr
import GMMs.segment_axis
np.set_printoptions(threshold = 1e6)     #  threshold表示输出数组的元素数目
import scipy.signal as signal
import librosa
from python_speech_features import mfcc,logfbank,get_filterbanks,delta
from matplotlib import pyplot as plt
from scipy.fftpack import dct
import librosa.display
import GMMs.zcr

def wavread(filename):
    f = wave.open(filename,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    f.close()
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    return waveData

class PreProcess():
    y=[]            #音频
    sr=16000            #采样率
    nf=0            #帧的总长度
    nw = 256
    inc = 128
    file_path=""
    def __init__(self,file_path,sr=16000,ifPlot=False):
        self.file_path=file_path
        self.sampling(file_path=file_path,sr=sr,ifPlot=ifPlot)

    # 输入语音流采用16KHz采样
    def sampling(self,file_path,sr=16000,ifPlot=False):
        self.sr=sr
        self.y, self.sr = librosa.load(file_path,sr=self.sr)
        if ifPlot==True:
            print(self.sr,len(self.y))
            librosa.display.waveplot(self.y, sr=self.sr, alpha=0.4)
            #plt.plot(self.y)
            plt.title('After Sampling')
            plt.show()
        return self.y,self.sr

    # 预增强
    def pre_emphasis(self,pre_emphasis = 0.97,ifPlot=False):
        Enhanced_y = np.append(self.y[0], self.y[1:] - pre_emphasis * self.y[:-1])
        if ifPlot==True:
            print(self.sr,len(self.y))
            librosa.display.waveplot(self.y, sr=self.sr, alpha=0.4)
            #plt.plot(Enhanced_y)
            plt.title('After Pre_Emphasis')
            plt.show()
        self.y=Enhanced_y
        return self.y

    #   分帧 以256个采样点为一个音框单位（帧），以128为音框之间的重迭单位，对输入语音流进行分帧
    def enframe(self, nw=256, inc=128,ifPlot=False):
        '''分帧
        参数含义：
        signal:原始音频型号
        nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
        inc:相邻帧的间隔（同上定义）
        '''

        #self.y=wavread(self.file_path)[0]

        self.nw=nw
        self.inc=inc
        signal_length=len(self.y) #信号总长度
        print(signal_length)
        if signal_length <= self.nw: #若信号长度小于一个帧的长度，则帧数定义为1
            self.nf = 1
        else: #否则，计算帧的总长度
            self.nf = int(np.ceil((1.0*signal_length-self.nw+self.inc)/self.inc))

        pad_length=int((self.nf-1)*self.inc+self.nw) #所有帧加起来总的铺平后的长度
        zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
        pad_signal=np.concatenate((self.y,zeros)) #填补后的信号记为pad_signal
        indices=np.tile(np.arange(0,nw),(self.nf,1))+np.tile(np.arange(0,self.nf*self.inc,self.inc),(self.nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
        indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
        self.y=pad_signal[indices] #得到帧信号


        if ifPlot==True:
            print(len(self.y))
            _shape=np.shape(self.y)
            print(_shape)
            '''
            a=len(self.y)
            sample_time = 1 / self.sr  # 采样点的时间间隔
            time = a / self.sr  # 声音信号的长度
            x_seq = np.arange(0, time, sample_time)
            '''
            '''
            for j in range(0,_shape[1]):
                y = []
                for i in range(0,_shape[0]):
                    y.append(self.y[i][j])
                y=np.array(y)
                print(y)
                librosa.display.waveplot(y, sr=self.sr, alpha=0.4)
            '''
            plt.plot(self.y)

            plt.title('After Enframe')
            plt.show()
        return self.y  #返回帧信号矩阵

    # 汉明窗 对每帧信号加窗
    def force_window(self,ifPlot=False):
        if self.nf==0:
            self.y,self.nf = self.enframe(self.y, self.nw, self.inc)
        win = np.tile(np.hamming(self.nw), (self.nf, 1))  # window窗函数，这里默认取1
        self.y = self.y * win
        if ifPlot == True:
            #librosa.display.waveplot(self.y, sr=self.sr, alpha=0.4)
            plt.plot(self.y)
            plt.title('After Hamming')
            plt.show()
        return  self.y

    # 提取MFCC和滤波器组特征
    def get_mfcc(self,ifPlot=False):

        feat_mfcc = mfcc(self.y, self.sr,)
        feat_mfcc_d = delta(feat_mfcc, 1)   # 一阶差分
        feat_mfcc_dd = delta(feat_mfcc, 2)  # 二阶差分
        mfcc_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))

        feat_fbank = logfbank(self.y, self.sr)
        feat_fbank_d = delta(feat_fbank, 1)   # 一阶差分
        feat_fbank_dd = delta(feat_fbank, 2)  # 二阶差分
        filterbank_feature = np.column_stack((feat_fbank, feat_fbank_d, feat_fbank_dd))

        print('\nMFCC:\n窗口数 =', mfcc_feature.shape[0])
        print('每个特征的长度 =', mfcc_feature.shape[1])
        print('\nFilter bank:\n窗口数 =', filterbank_feature.shape[0])
        print('每个特征的长度 =', filterbank_feature.shape[1])

        if ifPlot == True:
            # 画出特征图，将MFCC可视化。转置矩阵，使得时域是水平的
            mfcc_feature = mfcc_feature.T
            plt.matshow(mfcc_feature)
            plt.title('MFCC')
            plt.show()
            # 将滤波器组特征可视化。转置矩阵，使得时域是水平的
            filterbank_feature = filterbank_feature.T
            plt.matshow(filterbank_feature)

            plt.title('Filter bank')

            plt.show()
        self.y=mfcc_feature
        return self.y

    # 自定义函数，计算数值的符号。
    def sgn(self,data):
        if data >= 0:
            return 1
        else:
            return 0

    # 计算过零率
    def calZeroCrossingRate(self,ifPlot=False):
        zeroCrossingRate = []
        sum = 0
        for i in range(len(self.y)):
            if i % 256 == 0:
                continue
            sum = sum + np.abs(self.sgn(self.y[i]) - self.sgn(self.y[i - 1]))
            if (i + 1) % 256 == 0:
                zeroCrossingRate.append(float(sum) / 255)
                sum = 0
            elif i == len(self.y) - 1:
                zeroCrossingRate.append(float(sum) / 255)
        if ifPlot==True:
            plt.plot(zeroCrossingRate)
            plt.title('st-Zero Cross Rate')
            plt.show()
        return zeroCrossingRate
    # Normalising the spectral centroid for visualisation

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def Xdb (y1):
    ''' 频谱图'''
    X = librosa.stft(y1.y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=y1.sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

def Spectral_Centroids(y1):
    '''频谱质心'''
    spectral_centroids = librosa.feature.spectral_centroid(y1.y, sr=y1.sr)[0]
    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    # Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(y1.y, sr=y1.sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='r')
    plt.show()


def MFCC(y1,save_dir=None):
    '''MFCC '''
    mfccs = librosa.feature.mfcc(y1.y, sr=y1.sr)
    # Displaying  the MFCCs:
    librosa.display.specshow(mfccs, sr=y1.sr, x_axis='time')

    if save_dir!=None:
        plt.savefig(save_dir)

def Spectral_Rolloff(y1):
    '''光谱衰减'''
    spectral_centroids = librosa.feature.spectral_centroid(y1.y, sr=y1.sr)[0]
    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    spectral_rolloff = librosa.feature.spectral_rolloff(y1.y + 0.01, sr=y1.sr)[0]
    librosa.display.waveplot(y1.y, sr=y1.sr, alpha=0.4)
    plt.plot(t, normalize(spectral_rolloff), color='r')
    plt.show()


if __name__=="__main__":
    np.set_printoptions(threshold=1e6)  # threshold表示输出数组的元素数目

    file_path='霸王别姬.wav'
    #file_path = '测试.wav'
    y1=PreProcess(file_path)


    #ifPlot=True
#########################################预处理
    y1.y,y1.sr=y1.sampling(file_path,ifPlot=True,sr=None) #采样
    y1.pre_emphasis( ifPlot=True )       #预增强

    #####
    y1.enframe(nw=256,inc=128, ifPlot=True )            #分帧
    y1.force_window( ifPlot=True )       #加窗

#########################################特征提取
    file_path='霸王别姬.wav'
    '''   '''
    wave_data= zcr.load_wave_data(file_path)
    zcr.calEnergy(wave_data)  # 短时平均能量
    zeroCrossingRate = zcr.calZeroCrossingRate(wave_data)   #短时平均过零率


    #file_path = '测试.wav'
    y1.sampling(file_path, sr=None)

    Xdb(y1)               #频谱图
    '''
    Spectral_Centroids(y1)      #频谱质心
    '''
    plt.subplot(1, 1, 1)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(5.12,5.12)  # 输出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    MFCC(y1)                      #MFCC
    #Spectral_Rolloff(y1)














    # y1.get_mfcc( ifPlot=True )            #得mfcc
    #y1.calZeroCrossingRate(ifPlot=True)
    '''
    zcr=librosa.feature.zero_crossing_rate(y1.y)
    print(np.shape(zcr))
    print(zcr)
    plt.plot(zcr)
    plt.title("zero crossing rate")
    plt.show()
    '''
    # Plot the signal:
    '''短时过零率
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y1.y, sr=y1.sr)
    # Zooming in
    n0 = 9000
    n1 = 9100
    plt.figure(figsize=(14, 5))
    plt.plot(y1.y[n0:n1])
    plt.grid()
    plt.show()
    zero_crossings = librosa.zero_crossings(y1.y[n0:n1], pad=False)
    print(sum(zero_crossings))
    plt.plot(zero_crossings)
    '''

