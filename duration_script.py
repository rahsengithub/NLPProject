# This file will look into the folder directory and check duration of the .wav files
# If duration is greater than 2 minutes we will crop it

import wave, os
import contextlib
from pydub import AudioSegment
folder_path = "big_data/LibriSpeech/train-clean-100/"

extenstion = ".wav"

for subdir, dirs, files in os.walk(folder_path):
    #print(files)

    if not files:
        continue
    else:
        #print(subdir)
        wavfilename=subdir.split('/')[-2:]
        wavfilename='-'.join(wavfilename)
        wavfilename=wavfilename+'.wav'
        #print(wavfilename)
        home = os.getcwd()
        folder = home + "/" + subdir + "/" + wavfilename
        try:
            wav_audio = AudioSegment.from_wav(folder)
        except:
            print(folder)
            continue
            a = input()
        dur = len(wav_audio)
        #print(dur)
        if dur < 120000:
            name, wav = folder.split(".")
            path = name + "_1." + wav
            wav_audio.export(path, format = 'wav')
            continue
        else:
            sec = 120000
            start = 0
            while start < dur:
                slice = wav_audio[start:start+sec]
                start += sec
                name, wav = folder.split(".")
                path = name +"_"+str(int(start/sec))+"."+wav
                slice.export(path, format='wav')
            #last = wav_audio[start:dur]
            #c = str(int((start/sec)+1))
            #last.export(name+"_"+c+"."+wav, format='wav')


