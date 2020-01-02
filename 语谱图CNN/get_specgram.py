import csv
import sys

import math
import numpy as np
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold = 1e6)     #  threshold表示输出数组的元素数目
import wave
import matplotlib.mlab as mlab
import matplotlib
#import librosa

'''宏定义'''
if_plot=0  #是否绘制语谱图
ans=np.zeros([1,512])          #当前文件夹所有数据的标注都是ans
numFeatures=1           #每组音频的特征数量，为1则表示二分类
ans[0][0]=1.0               #表示数据标注为1.0
fill=-150
abandon=True

'''生成语谱图'''
def specgram(x, NFFT=None, Fs=None, Fc=None, detrend=None,
             window=None, noverlap=None,
             cmap=None, xextent=None, pad_to=None, sides=None,
             scale_by_freq=None, mode=None, scale=None,
             vmin=None, vmax=None, **kwargs):
    if NFFT is None:
        NFFT = 256  # same default as in mlab.specgram()
    if Fc is None:
        Fc = 0  # same default as in mlab._spectral_helper()
    if noverlap is None:
        noverlap = 128  # same default as in mlab.specgram()
    if mode == 'complex':
        raise ValueError('Cannot plot a complex specgram')
    if scale is None or scale == 'default':
        if mode in ['angle', 'phase']:
            scale = 'linear'
        else:
            scale = 'dB'
    elif mode in ['angle', 'phase'] and scale == 'dB':
        raise ValueError('Cannot use dB scale with angle or phase mode')
    spec, freqs, t = mlab.specgram(x=x, NFFT=NFFT, Fs=Fs,
                                   detrend=detrend, window=window,
                                   noverlap=noverlap, pad_to=pad_to,
                                   sides=sides,
                                   scale_by_freq=scale_by_freq,
                                   mode=mode)
    if scale == 'linear':
        Z = spec
    elif scale == 'dB':
        if mode is None or mode == 'default' or mode == 'psd':
            Z = 10. * np.log10(spec)
        else:
            Z = 20. * np.log10(spec)
    else:
        raise ValueError('Unknown scale %s', scale)
    Z = np.flipud(Z)
    if xextent is None:
        # padding is needed for first and last segment:
        pad_xextent = (NFFT - noverlap) / Fs / 2
        xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]
    im = plt.imshow(Z, cmap, extent=extent, vmin=vmin, vmax=vmax,
                     **kwargs)
    plt.axis('auto')
    return spec, freqs, t, im,Z


class spec_cnn:
    fileFolder=""
    dataFolder="dataset"
    wave_data=[]
    row=512
    col=512

    def __init__(self,ff):
        self.fileFolder=ff

    '''加载wav文件'''
    def load_one_data(self,path):
        # 打开语音文件
        f=wave.open(path,'rb')
        # 得到语音参数
        params=f.getparams()
        nchannels,samp_width,frame_rate,nframes=params[:4]
        #将字符串格式的数据转成int型
        str_data=f.readframes(nframes)
        self.wave_data=np.fromstring(str_data,dtype=np.short)
        #归一化
        self.wave_data=self.wave_data*1.0/max(abs(self.wave_data))
        #将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
        self.wave_data=np.reshape(self.wave_data,[nframes,nchannels]).T
        f.close()
        return self.wave_data,nchannels,samp_width,frame_rate,nframes


    '''生成一个wav文件的语谱图'''
    def get_one_spec(self,path,pl=False):
        self.wave_data,nchannels,sampwidth,frame_rate,nframes=self.load_one_data(path)
        frame_length=0.025  #帧长20~30ms
        frame_size=frame_length*frame_rate#每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等
                                        # 而NFFT最好取2的整数次方,即framesize最好取的整数次方
        # 找到与当前framesize最接近的2的正整数次方
        nfft_dict={}
        lists=[32,64,128,256,512,1024]
        for i in lists:
            nfft_dict[i]=abs(frame_size-i)
        sort_list=sorted(nfft_dict.items(),key=lambda x: x[1])#按与当前framesize差值升序排列
        frame_size=int(sort_list[0][0]) #取最接近当前framesize的那个2的正整数次方值为新的framesize
        NFFT=frame_size  #NFFT必须与时域的点数framsize相等，即不补零的FFT
        overlap_size = 1.0 / 3 * frame_size  # 重叠部分采样点数overlap_size约为每帧点数的1/3~1/2
        overlap_size = int(round(overlap_size))  # 取整
        # 绘制频谱图
        #spectrum,freqs,ts,fig=plt.specgram(self.wave_data[0],NFFT=NFFT,Fs=frame_size,window=np.hanning(M = frame_size),noverlap=overlap_size,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)
        spectrum, freqs, ts, fig,RGB = specgram(self.wave_data[0], NFFT=NFFT, Fs=frame_rate,
                                                window=np.hanning(M=frame_size),
                                                noverlap=overlap_size, mode='default', scale_by_freq=True,
                                                sides='default',
                                                scale='dB', xextent=None)  # 绘制频谱图

        print("帧长为{},帧叠为{},傅里叶变换点数为{}".format(frame_size, overlap_size, NFFT))
        print("语谱图大小：", np.shape(spectrum))
        print("图像矩阵大小:",np.shape(RGB))
        plt.savefig(str(0) + ".png")
        if(pl):
            print("绘制语谱图...")
            plt.ylabel('Frequency')
            plt.xlabel('Time(s)')
            plt.title('Spectrogram')
            plt.show()
        return spectrum,freqs,ts,fig,RGB

    def save_one_data(self,RGB,filename):
        filename=filename[:-3]+'csv'
        path=os.path.join(self.fileFolder,self.dataFolder)
        if os.path.exists(path)==False:
            os.mkdir(path)
        with open(os.path.join(path,filename),'w',newline='') as file:
            np.savetxt(os.path.join(path,filename), RGB, delimiter=',')
        with open(os.path.join(path,filename), "a+", newline='') as file:
            csv_file = csv.writer(file)
            csv_file.writerows(ans)


    '''读取所有数据集，或读取一条数据集'''
    def read_data(self,path=None,one_path=None):
        if one_path==None:
            data_list=[]
            if path == None:
                path = os.path.join(self.fileFolder, self.dataFolder)
            one_data = []
            if os.path.exists(path) == False:
                print(path,"无该文件夹！")
            flag=False
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                print("读取文件：", file_path, "...")
                one_data = np.loadtxt(open(file_path), delimiter=",", skiprows=0)
                data_list.append(one_data)
                # print("data:",one_data)
                print("data大小：", np.shape(one_data))
                # print("data类型：",type(one_data))
                if flag==False :
                    flag=True
            if flag==False:
                print(path,"文件夹为空！")
            else:
                return data_list
        else:
            if os.path.exists(one_path):
                one_data = np.loadtxt(open(one_path), delimiter=",", skiprows=0)
                return one_data
            else:
                print(one_path,"文件不存在！")


    '''生成一个文件夹的所有wav文件的语谱图'''
    def get_spec(self):
        emp = []
        for i in range(0, self.col):
            emp.append(fill)
        for file in os.listdir(self.fileFolder):
            print(file)
            if file[-3:]!="wav":
                continue
            file_path=os.path.join(self.fileFolder,file)
            print("正在生成",file,"的语谱图...")
            spectrum, freqs, ts, fig,RGB=self.get_one_spec(file_path,if_plot)
            #print("图像矩阵：",RGB)
            ###########################################################################################
            ''''''
            if abandon and (np.shape(RGB)[0] < 480 or np.shape(RGB)[1]<480):
                print(file,np.shape(RGB),"大小离512*512太远，舍弃！")
                continue

            '''是否将数据裁剪'''
            print("数据裁剪为",self.row,"x",self.col,"...")
            res=[]
            now=self.row
            maxR=self.row
            if np.shape(RGB)[0] < self.row:
                now=self.row-np.shape(RGB)[0]
                print(now)
                for i in range(0,now):
                    res.append(emp)
            maxR=min(np.shape(RGB)[0],maxR)

            for i in range(0,maxR):
                if np.shape(RGB)[1]<self.col:
                    temp=RGB[i].tolist()
                    while(np.size(temp)<self.col):
                        temp.append(fill)#float("-inf"))
                else:
                    temp=RGB[i].tolist()[0:self.col]
                res.append(temp)
            res=np.array(res)
            print("裁剪后图像矩阵大小：",np.shape(res))

            '''替换+-inf'''
            for i in range(0,self.row):
                for j in range(0,self.col):
                    if res[i][j]==float("-inf") or res[i][j]== float("inf") or math.isnan(res[i][j]):
                        res[i][j]=fill

            if if_plot==True:
                plt.imshow(res)
                plt.ylabel('Freq')
                plt.show()
            RGB=res

            '''将生成的语谱图的图像矩阵存到文件中作为数据集'''
            self.save_one_data(RGB,file)

def get_dataset(train_name='train_wav',test_name='test_wav',num_feature=numFeatures):
    train_ = spec_cnn(train_name)
    test_ = spec_cnn(test_name)
    train_data = train_.read_data()
    test_data = test_.read_data()
    train_label = []
    test_label = []
    for i in range(0,np.shape(train_data)[0]):
        temp=train_data[i][512][0:num_feature]
        train_label.append(temp.tolist())
        train_data[i]=train_data[i][0:512]
    for i in range(0,np.shape(test_data)[0]):
        temp=test_data[i][512][0:num_feature]
        test_label.append(temp.tolist())
        test_data[i] = test_data[i][0:512]
    return train_data,train_label,test_data,test_label

if __name__ == "__main__":
    '''终端输出保存到文件
    output = sys.stdout
    outputfile = open("output.txt", "w+")
    sys.stdout = outputfile
    #type = sys.getfilesystemencoding()  # python编码转换到系统编码输出
    '''
    train_ = spec_cnn('train_wav')
    test_ = spec_cnn('test_wav')
    train_.get_spec()
    test_.get_spec()
    '''
    train_data, train_label, test_data, test_label=get_dataset()
    print(train_label)
    print(test_label)
    print(np.shape(train_data))
    print(np.shape(test_data))
    '''



















    '''
    fileFolder='train_wav'
    cnn=spec_cnn(fileFolder)
    '''
    '''
    cnn.get_spec()
    cnn.read_data()
    temp=cnn.read_data(one_path="train_wav\\dataset\\01 曲目 1 48000 1_1.csv")
    print("temp:",np.shape(temp))
    '''

    '''测试语谱图
    spectrum,freqs,ts,fig,RGB=cnn.get_one_spec("01.wav",True)
    print(spectrum)
    print(RGB)
    print(freqs)
    '''
