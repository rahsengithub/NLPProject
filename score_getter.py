# Get word-based variables and sound based variables in same python script

# word-based, load JSON file
import json
from statistics import mean
import os

folder_path = "jobfiles"
home = os.getcwd()
Score_dict = {}

for dirs, subdirs, files in os.walk(folder_path):
    json_name = dirs
    if len(files) == 0:
        continue
    json_file_name = files[0]

    path = home + "/" + json_name + "/" + str(json_file_name)

    with open(path) as f:
        data = json.load(f)

    # Get features from JSON dict
    Score = data["score"]
    id = data["id"]
    Score_dict[id] = Score



    # Extract features from JSON dicionary

    # all tokens
    tokens = data["tokens"]
    time = data["elapsed_time"] * (1 / 60)
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

    SIL_lengts = SIL_lengths[1:-1]
    SIL_lengths = [sil for sil in SIL_lengths if sil > 2.0]
    pause_freq = len(SIL_lengths) / time
    voc_freq = len(dist_list) / time
    dist_lengths = len("hei")

    voc_avg = mean([len(string) for string in dist_list])

f = open("dict_score.txt","w")
f.write(str(Score_dict))
f.close()

