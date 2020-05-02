# coding=gbk
import os
import wave
import numpy as np
import pylab as plt

'''
��Ƶ�ȼ���ָ�
'''

CutTimeDef = 8  # ��8s�ض��ļ�
path = r"train_no"

files = os.listdir(path)
files = [path + "\\" + f for f in files if f.endswith('.wav')]

def CutFile():
    for i in range(len(files)):
        FileName = files[i]
        print("CutFile File Name is ", FileName)
        f = wave.open(r"" + FileName, "rb")
        params = f.getparams()
        print(params)
        nchannels, sampwidth, framerate, nframes = params[:4]
        CutFrameNum = framerate * CutTimeDef
        # ��ȡ��ʽ��Ϣ
        # һ���Է������е�WAV�ļ��ĸ�ʽ��Ϣ�������ص���һ����Ԫ(tuple)��������, ����λ����byte    ��λ��, ��
        # ��Ƶ��, ��������, ѹ������, ѹ�����͵�������waveģ��ֻ֧�ַ�ѹ�������ݣ���˿��Ժ������������Ϣ

        print("CutFrameNum=%d" % (CutFrameNum))
        print("nchannels=%d" % (nchannels))
        print("sampwidth=%d" % (sampwidth))
        print("framerate=%d" % (framerate))
        print("nframes=%d" % (nframes))
        str_data = f.readframes(nframes)
        f.close()  # ����������ת��������

        # ��Ҫ������������������λ������ȡ�Ķ���������ת��Ϊһ�����Լ��������
        wave_data = np.fromstring(str_data, dtype=np.short)
        wave_data.shape = -1, 2
        wave_data = wave_data.T
        temp_data = wave_data.T

        StepNum = CutFrameNum
        StepTotalNum = 0;
        haha = 0
        while StepTotalNum < nframes:

            print("Stemp=%d" % (haha))
            FileName =  files[i][0:-4] + "-" + str(haha + 1) + ".wav"
            print(FileName)
            temp_dataTemp = temp_data[StepNum * (haha):StepNum * (haha + 1)]
            haha = haha + 1;
            StepTotalNum = haha * StepNum;
            temp_dataTemp.shape = 1, -1
            temp_dataTemp = temp_dataTemp.astype(np.short)

            f = wave.open(FileName, "wb")  # ��WAV�ĵ�

            # ����������������λ����ȡ��Ƶ��
            f.setnchannels(nchannels)
            f.setsampwidth(sampwidth)
            f.setframerate(framerate)

            # ��wav_dataת��Ϊ����������д���ļ�
            f.writeframes(temp_dataTemp.tostring())
            f.close()


if __name__ == '__main__':
    CutFile()
    print("Run Over")
