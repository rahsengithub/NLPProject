
import os
from pydub import AudioSegment
folder_path = "big_data/LibriSpeech/train-clean-100/"
folder_path_exp = "/big_data/LibriExtract/"

for subdir, dirs, files in os.walk(folder_path):
    if not files:
        continue
    else:
        string = "_"
        home = os.getcwd()
        folder = home + "/" + subdir + "/"
        for f in files:
            if string in f:
                wav_audio = AudioSegment.from_wav(folder+f)
                wav_audio.export(home+folder_path_exp+f, format='wav')

