import tarfile
import os

folder = "jobFiles"
new_folder = "data_folder"
home = os.getcwd()
full_path = home + new_folder
for subdirs, dirs, files in os.walk(folder):
    for file in files:
        #print(file)
        new_path = os.path.join(os.path.join(home,new_folder))
        #new_path = 'U:\Team_Project\jobFiles\data_folder'
        #print(new_path)
        
        tar = tarfile.open(os.path.join(subdirs,file))
        tar.extractall(path=new_path)
        tar.close()
        
    