import wave
import numpy as np
from matplotlib import pyplot as plt

# 自定义函数，计算数值的符号。
def sgn(data):
    if data >= 0 :
        return 1
    else :
        return 0
#计算过零率
def calZeroCrossingRate(wave_data) :
    zeroCrossingRate = []
    sum = 0

    for i in range(len(wave_data)) :
        #
        if i % 256 == 0:
            #print(i)
            continue
        sum = sum + np.abs(sgn(wave_data[i]) - sgn(wave_data[i - 1]))
        if (i + 1) % 256 == 0 :
            zeroCrossingRate.append(float(sum) / 255)
            sum = 0
        elif i == len(wave_data) - 1 :
            zeroCrossingRate.append(float(sum) / 255)
    print("done")
    plt.plot(zeroCrossingRate)
    plt.title('st-Zero Cross Rate')
    plt.show()
    return zeroCrossingRate

# 计算每一帧的能量 256个采样点为一帧
def calEnergy(wave_data) :
    energy = []
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % 256 == 0 :
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1 :
            energy.append(sum)
    plt.plot(energy)
    plt.title('Short Energy')
    plt.show()
    return energy

def load_wave_data(file_path):
    f = wave.open(file_path, "rb")
    # getparams() 一次性返回所有的WAV文件的格式信息
    params = f.getparams()
    # nframes 采样点数目
    nchannels, sampwidth, framerate, nframes = params[:4]
    # readframes() 按照采样点读取数据
    str_data = f.readframes(nframes)  # str_data 是二进制字符串

    # 以上可以直接写成 str_data = f.readframes(f.getnframes())

    # 转成二字节数组形式（每个采样点占两个字节）
    wave_data = np.fromstring(str_data, dtype=np.short)
    print("采样点数目：" + str(len(wave_data)))  # 输出应为采样点数目
    f.close()
    return  wave_data

if __name__=="__main__":

    file_path="霸王别姬.wav"
    wave_data=load_wave_data(file_path)
    calEnergy(wave_data)  # 短时平均能量
    zeroCrossingRate = calZeroCrossingRate(wave_data)   #短时平均过零率
