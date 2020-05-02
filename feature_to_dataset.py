from pretreat_to_picture import *
import os
import matplotlib.pyplot as plt
'''
将音频数据批量转换成训练级（MFCC谱图和语谱图）
'''
data_dir_list=[
    'data_audio/train/is/',
    'data_audio/train/not/',
    'data_audio/test/is/',
    'data_audio/test/not/',
]

def one_to_dataset(data_dir):
    # 对每一个文件夹里的音频文件进行处理
    for each in os.listdir(data_dir):
        if each[-3:] != "wav":                      # 如果不是wav文件则跳过
            continue
        y1 = PreProcess(data_dir+each)              # 实例化预处理模型
        plt.subplot(1, 1, 1)                        # 设置绘图参数
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(2.56,2.56)  # 输出width*height像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        MFCC(y1)                                    # 绘制MFCC谱图

        save_dir = "data_set/"+"mfcc"+data_dir[10:]  # 保存MFCC谱图
        plt.savefig(os.path.join(save_dir,(each[:-3]+"png")))

        Xdb(y1,True)                               # 绘制语谱图
        save_dir = "data_set/"+"specgram"+data_dir[10:]   # 保存语谱图
        plt.savefig(os.path.join(save_dir,(each[:-3]+"png")))

        plt.close('all')


def main():
    for each in data_dir_list:
        one_to_dataset(each)

if __name__ == "__main__":
    main()