# This file will look into the folder directory and check duration of the .wav files
# If duration is greater than 2 minutes we will crop it

import wave, os
import contextlib
from pydub import AudioSegment
folder_path = "TEDLIUM_release-3/data"
old_path = folder_path+"/wav"
new_path = folder_path+"/wav_splitted"
home = os.getcwd()


counter=0
for files in os.listdir(old_path):
    full = home+"/"+old_path+"/"+files
    try:
        wav_audio = AudioSegment.from_wav(full)
    except:
        print(files)
        continue
        a = input()
    wav_audio = wav_audio[15000:]
    dur = len(wav_audio)
    if dur < 120000:
        name, wav = files.split(".")
        new = name + "_1." + wav
        export_path = home + "/"+ new_path+ "/"+ new
        wav_audio.export(export_path, format = 'wav')
        continue
    else:
        sec = 120000
        start = 0
        while start < dur:
            slice = wav_audio[start:start+sec]
            start += sec
            name, wav = files.split(".")
            new = name +"_"+str(int(start/sec))+"."+wav
            export_path = home + "/"+ new_path+ "/"+new
            slice.export(export_path, format='wav')
    print ("I have completed "+str(counter)+ " wav files")
    counter+=1



