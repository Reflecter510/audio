import wave
from pydub import AudioSegment
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 想遍历的文件夹的路径
'''需要事先在文件夹下新建pic_one和cut_pic目录，不然会报错'''
path = r'./is_4s/'
filepath='./is_4s/'

def pic_cut(picpath):
	print('picpath:',picpath)
	#将频谱图全部分割为256*256
	img = Image.open(picpath)
	#print(img.width/256)
	for i in range(0,int(img.width/80)):
		cpic_path = filepath+"cut_pic/"+os.path.split(picpath)[1][:-4]+"("+str(i)+")"+".png"
		print('cpic_path:',cpic_path)
		print('int(img.width/80):',int(img.width/80))
		cropped = img.crop((i*80,0,i*80+80,256))
		cropped.save(cpic_path)

a=[]


'''
for files in os.listdir(path):
    print(files)
    source_file_path = path + files
    destin_path = path + '\\' + files[:-3] + 'wav'
    sound = AudioSegment.from_wav(source_file_path)
    sound.export(destin_path, format='wav')
'''


for files in os.listdir(filepath):
    print('filename:', files)
    wav_path = filepath + files
    pic_path = filepath+'pic_one/' + files[:-3] + 'png'
    if wav_path[-3:]!='wav':
        continue
    f = wave.open(wav_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print('nchannels(声道数):', nchannels)
    print('sampwidth(量化位数byte):', sampwidth)
    print('framerate(采样频率):', framerate)
    print('nframes(采样点数):', nframes)

    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    print('waveData:', waveData)
    # 多通道处理
    waveData = np.reshape(waveData, [nframes, nchannels])
    f.close()  # 关闭文件读写流

    time = np.arange(0, nframes) * (1.0 / framerate)
    print('time:', time)

    # 绘图，两个声道分别画
    plt.figure()
    # #第一个声道
    # plt.subplot(7,1,1)
    # plt.plot(time,waveData[:,0])
    # plt.xlabel("Time(s)")
    # plt.ylabel("Amplitude")
    # plt.title("Ch-1 wavedata")
    # plt.grid(True)#标尺,on:有,off:无

    # #第二个声道
    # plt.subplot(7,1,3)
    # plt.plot(time,waveData[:,1])
    # plt.xlabel("Time(s)")
    # plt.ylabel("Amplitude")
    # plt.title("Ch-2 wavedata")
    # plt.grid(True)#标尺,on:有,off:无

    # 频谱图（横坐标时间、纵坐标频率）
    # 第一声道
    plt.subplot(1, 1, 1)
    plt.specgram(waveData[:, 0], Fs=framerate, scale_by_freq=True, sides='default')
    # plt.ylabel('Frequency(Hz)')
    # plt.xlabel('Time(s)')
    # #第二声道
    # plt.subplot(3,1,3)
    # plt.specgram(waveData[:,1],Fs = framerate,scale_by_freq = True,sides='default')
    # plt.ylabel('Frequency(Hz)')
    # plt.xlabel('Time(s)')

    plt.axis('off')

    fig = plt.gcf()
    width = nframes / framerate / 10
    height = 2.56
    fig.set_size_inches(width, height)  # 输出width*height像素

    fig.set_size_inches(2.56,2.56)  # 输出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    #plt.show()
    plt.savefig(pic_path)

    plt.close('all')  # 防止内存溢出
    #pic_cut(pic_path)

