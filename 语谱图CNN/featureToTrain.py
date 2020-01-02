from GMMs.PretreatMFCC import *
import os
import matplotlib.pyplot as plt
data_dir='./is_4s/'
save_dir='./is_4s/is_mfcc/'

for each in os.listdir(data_dir):
    if each[-3:] != "wav":
        continue
    y1 = PreProcess(data_dir+each)
    plt.subplot(1, 1, 1)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(2.56,2.56)  # 输出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    MFCC(y1)                      #MFCC
    plt.savefig(os.path.join(save_dir,(each[:-3]+"png")))
    #plt.show()
    plt.close('all')