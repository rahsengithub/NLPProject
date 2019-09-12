import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import os
path='/home/ibbu/Desktop/MSC_Data_analytics/industrial_team_project/Projects-group7-gweek/Deep_Learning/train/Audio-Books'
all_files=[os.path.join(path, x) for x in os.listdir(path)]


path='/home/ibbu/Desktop/MSC_Data_analytics/industrial_team_project/Projects-group7-gweek/Deep_Learning/train/TED'
all_files_TED=[os.path.join(path, x) for x in os.listdir(path)]

path_audiobooks='/home/ibbu/Desktop/MSC_Data_analytics/industrial_team_project/Projects-group7-gweek/Deep_Learning/train/Audio-Books'

def request_function(filepath):
	filename=os.path.basename(filepath)
	m = MultipartEncoder(fields={'CustomData': '','file': (filename, open(filepath, 'rb'), 'audio/wav')})
	r = requests.post('https://admin.gweekspeech.com/api/v1/analysis', data=m,
                      headers={'Content-Type': m.content_type,  'Authorization': "Bearer lljXCMX3PHZNvrf8eApNvDtF4rSMdMKcSmUe0Ivtu6e0OT6tDCbQXqQ63XxD-9IlxYaDivYqTTGCqk3LjmSgB2rZJuz9-2EjPlwzYgf0LkM3S4kWDD29BEiQJkMFa3-FFrANI6cB_Y7SPdF4d1nv4ZpQpsNn0cjw8hC4_Heu0IREJk7aGiY_vEecPp_iksm9W3Ew-ZNpem7M0r90-QKcRLb28QQlFoe5QeohrvlsRkcfBK_qhoU-HXWXfMmewVihpXpL1w3IZpPv4yB5dLX2z2rth88BDM3pTYNU22KCVn3vToaO4rvw7imzi5dtwm53PMoJCLNQl33qyH55wp6YCX3ORi_U5WlBWUSfrKPYb5nGn08DqwPImCQ2ddg5_xsSTgi9_SjCNB9vwzXKTox6iGz-Ro8TRHa7y_B7MzppZwOPJJq634Iv1qHOSjMZO5Vl"})

	return r.text
	
from random import shuffle

def get_containing_data(filename):
	f=open(filename,'r')
	data=f.read().split('\n')
	f.close()
	data=data[1:(len(data)-1)]
	shuffle(data)
	f=open('audio_references_fixed.csv','r')
	sd=f.read().split('\n')
	sd=sd[1:(len(sd)-1)]
	flist=give_me_fname(sd)

	
	print ("function started")
	for x in data:
		print (x)
		all_f=x.split('____')
		ftype=all_f[0]
		filename1=all_f[1]
		if filename1 in flist:
			continue
		fid=all_f[2]
		if fid.find('API calls')!=-1:
			newfilename=''
			if ftype=='TED':
				newfilename=os.path.join(path, filename1)
			else:
				newfilename=os.path.join(path_audiobooks,filename1)
			start=time.time()
			newid=request_function(newfilename)
			end=time.time()
			
			print (newfilename+','+newid+" it took "+str(end-start))
			f = open('audio_references_fixed2.csv', 'a+')
			f.write(ftype+'____'+os.path.basename(newfilename)+'____'+newid+'\n')
			f.close()
			time.sleep(120)
		else:
			newfilename=''
			if ftype=='TED':
				newfilename=os.path.join(path, filename1)
			else:
				newfilename=os.path.join(path_audiobooks,filename1)
			start=time.time()
			newid=request_function(newfilename)
			end=time.time()
			
			print (newfilename+','+newid+" it took "+str(end-start))
			f = open('audio_references_fixed.csv', 'a+')
			f.write(ftype+'____'+os.path.basename(newfilename)+'____'+newid+'\n')
			f.close()
			time.sleep(300)

		
	


#f.write('TYPE____filename____id\n')
#f.close()
import time


def give_me_fname(datalist):
	filelist=[]
	for x in datalist:
		all_f=x.split('____')
		ftype=all_f[0]
		filename=all_f[1]
		fid=all_f[2]
		filelist.append(filename)
	return filelist
f=open('audio_references.csv','r')
data=f.read().split('\n')
f.close()
data=data[1:len(data)-1]
f=open('audio_references_fixed.csv','r')
data1=f.read().split('\n')
data1=data1[1:len(data1)-1]
data=data+data1

f.close()

filelist=give_me_fname(data)
#get_containing_data('Problem_ids.csv')
#exit()
all_files_combined=all_files_TED+all_files
shuffle(all_files_combined)
shuffle(all_files_combined)
for x in all_files_combined:
	start = time.time()
	filename=os.path.basename(x)
	ftype=x.split('/')[-2]
	if ftype!='TED':
		ftype='AUDIOBOOKS'
	print (x,ftype)
	if filename in filelist:
		print ('file ' +filename+' already in filelist continuing')
		continue

	m = MultipartEncoder(fields={'CustomData': '','file': (filename, open(x, 'rb'), 'audio/wav')})
	r = requests.post('https://admin.gweekspeech.com/api/v1/analysis', data=m,
                      headers={'Content-Type': m.content_type,  'Authorization': "Bearer lljXCMX3PHZNvrf8eApNvDtF4rSMdMKcSmUe0Ivtu6e0OT6tDCbQXqQ63XxD-9IlxYaDivYqTTGCqk3LjmSgB2rZJuz9-2EjPlwzYgf0LkM3S4kWDD29BEiQJkMFa3-FFrANI6cB_Y7SPdF4d1nv4ZpQpsNn0cjw8hC4_Heu0IREJk7aGiY_vEecPp_iksm9W3Ew-ZNpem7M0r90-QKcRLb28QQlFoe5QeohrvlsRkcfBK_qhoU-HXWXfMmewVihpXpL1w3IZpPv4yB5dLX2z2rth88BDM3pTYNU22KCVn3vToaO4rvw7imzi5dtwm53PMoJCLNQl33qyH55wp6YCX3ORi_U5WlBWUSfrKPYb5nGn08DqwPImCQ2ddg5_xsSTgi9_SjCNB9vwzXKTox6iGz-Ro8TRHa7y_B7MzppZwOPJJq634Iv1qHOSjMZO5Vl"})

	end=time.time()
	print (filename+','+r.text+" it took "+str(end-start))
	f = open('audio_references_fixed.csv', 'a+')
	
	f.write(ftype+'____'+filename+'____'+r.text+'\n')
	f.close()
	time.sleep(180)



exit()




for x in all_files[:500]:
    
	filename=os.path.basename(x)
	if filename in filelist:
		print ('file ' +filename+' already in filelist continuing')
		continue
	start = time.time()
	m = MultipartEncoder(
        fields={'CustomData': '',
                'file': (filename, open(x, 'rb'), 'audio/wav')}
        )

	r = requests.post('https://admin.gweekspeech.com/api/v1/analysis', data=m,
                      headers={'Content-Type': m.content_type,  'Authorization': "Bearer Z2p7MGiOKXV52SUlcgrb8zn0uOQWTa3hJWxLX7RtmtYNg2ra1lqRhEwf--_p6CiLlfDafTKtRv3ADEottDTTAfP0tgt_zvHfSEqbAQo8FzO38s-LlFT8GVyP6Sesw2hQWnfOMhabf6HPcmuW3owXZQaRpzJEPhriC8bbb10vtOmFjrippm78TT1Ly68qIR52GxAlvWwFvSuy0fC_1A0fhUAnpk2UNQ8SciVvePHrEZsOC7--SAQCEGytncQVisXcCJMSYQDaUb9acsiyMUQDf84oAohz5FlYXxN_IyWvNfuw1U25daqXiQ0U4m4dax4Sm_eK5wGSF2bzYQzMMMaPlXGKGKkl5YBooEOBjby8CdqGwEgqBzRWUnjT4IdmIr1-FhZfrK4l8Fmw-OU-_JxFcEYb23q2KcZOfuDMP1pEW1Eh2wOnp0BtTyeKGGbS7TPe"})

	end=time.time()
	print (filename+','+r.text+" it took "+str(end-start))
	f = open('audio_references_fixed.csv', 'a+')
	f.write('AUDIOBOOKS____'+filename+'____'+r.text+'\n')
	f.close()
	time.sleep(120)





