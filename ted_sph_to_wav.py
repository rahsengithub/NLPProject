from sphfile import SPHFile

import os
filename="ted_smallfile"

current_path=os.getcwd()

file_path=os.path.join(current_path, filename)
file_path=os.path.join(file_path,'data')

sph_path=os.path.join(file_path,'sph')
wav_file_path=os.path.join(file_path,'wav')
print (wav_file_path)

if not os.path.exists(wav_file_path):
    print (wav_file_path+" Not present creating a directory")
    os.makedirs(wav_file_path)
    print ("Created successfully now running")


report_number=100
len_of_files=len(os.listdir(sph_path))
for i,x in enumerate(os.listdir(sph_path)):
    if i%report_number==0: #for keeping the track


        percentage=float(i)/len_of_files
        percentage*=100
        print (str(i) +" out of "+str(len_of_files)+" completed which is: "+str(percentage)+"%")
    full_audio_path=os.path.join(sph_path,x)
    sph=SPHFile(full_audio_path)
    wav_filename=x[:x.find('.')]+'.wav'
    full_wav_file_path=os.path.join(wav_file_path,wav_filename)
    sph.write_wav(full_wav_file_path)

