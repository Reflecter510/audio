import os
import ffmpeg
from pydub import AudioSegment

source_file_path="./MeiLanFang1/"
destin_path="./Meilanfang/"
for file in os.listdir(source_file_path):
    file_path = os.path.join(source_file_path, file)
    print(file_path)
    sound = AudioSegment.from_mp3(file_path)
    dest=os.path.join(destin_path, file)
    sound.export(dest[0:-3]+"wav",format ='wav')
