import json
from statistics import mean
from statistics import stdev
import os

folder_path = "jobfiles"
home = os.getcwd()
voc_dict = {}

for dirs, subdirs, files in os.walk(folder_path):
    json_name = dirs
    if len(files) == 0:
        continue
    json_file_name = files[0]

    path = home + "/" + json_name + "/" + str(json_file_name)

    with open(path) as f:
        data = json.load(f)


    # Extract features from JSON dicionary

    # all tokens
    tokens = data["tokens"]
    time = data["elapsed_time"] * (1 / 60)
    id = data["id"]
    # get distinct words
    dist_list = []
    SIL_lengths = []
    for tok in tokens:
        new_tok = tok["text"]
        if new_tok == "SIL":
            SIL_lengths.append(tok["duration"])
        else:
            if new_tok not in dist_list:
                dist_list.append(new_tok)
            else:
                continue


    voc_freq = len(dist_list) / time
    dist_lengths = len("hei")

    voc_avg = mean([len(string) for string in dist_list])
    voc_std = stdev([len(string) for string in dist_list])

    voc_dict[id] = {"frequency:":voc_freq, "average length": voc_avg, "std deviation": voc_std}



f = open("vocabulary.txt","w")
f.write(str(voc_dict))
f.close()

