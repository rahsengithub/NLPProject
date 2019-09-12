from pydub import AudioSegment
import os
current_path=os.getcwd()

FILENAME=os.path.join("LibriSpeech","dev-clean")
print (FILENAME)


def print_chapter_content(chapter_path):
    all_files_in_path=sorted(os.listdir(chapter_path))
    print (all_files_in_path)
    #print ("Error can be here if they are not in sorted order")

    complete_audio=None
    for index,audio in enumerate(all_files_in_path):
        if audio.find('.flac')==-1:
            #print ("last thing to be printed")
            #print (index, len(all_files_in_path))
            #print ("______")
            continue
        full_filename=os.path.join(chapter_path, audio)
        audio_bin=AudioSegment.from_file(full_filename, format='flac')
        if complete_audio==None:
            complete_audio=audio_bin
        else:
            complete_audio+=audio_bin
    new_filename=all_files_in_path[0].split('-')[:-1]
    new_filename='-'.join(new_filename)
    new_filename+='.wav'
    full_path_new_filename=os.path.join(chapter_path, new_filename)
    complete_audio.export(full_path_new_filename,format='wav')

folder_path=os.path.join(current_path,FILENAME)
print (folder_path)
for speakers in os.listdir(folder_path):
    speaker_path=os.path.join(folder_path,speakers)
    for chapter in os.listdir(speaker_path):

        chapter_path=os.path.join(speaker_path,chapter)
        print_chapter_content(chapter_path)

