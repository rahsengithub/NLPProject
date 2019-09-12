# Import libraries
import os, json
import pandas as pd
from joblib import load
from sklearn.decomposition import TruncatedSVD


# Specify path to unlabelled files
path_to_json = '../data_folder/json_train'

# Load model
gnb = load('naive_bayes_model.joblib')


new_dict = {}
new_dict["id"] = []
new_dict["pause_duration"] = []
new_dict["no_of_pauses"] = []
new_dict["uh_duration"] = []
new_dict["no_of_uh"] = []
new_dict["um_duration"] = []
new_dict["no_of_um"] = []

pause_duration = []
pause_per_sec = []
setter = set()
inter = set()


# Loop through unlabeled files
id_list = []
for subdir, dirs, files in os.walk(path_to_json):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(subdir, file)

            with open(path, 'r') as f:
                json_text = json.load(f)

            # Get id using both folder name and id from JSON
            id_ = file

            feature_dict = {}

            score = json_text["score"]

            # Dont label if it is already labelled
            if id_ in id_list:
                continue

            id_list.append(id_)

            # Dont label JSONs with score < 5
            if score < 5:
                print("Outlier, Score less than 5")
                continue

            # Dont label JSONs without tokens
            if not json_text["tokens"]:
                print("Outlier, no tokens")
                continue


            new_dict["id"].append(id_)

            elapsed_time = json_text["elapsed_time"]
            counter_pauses = 0
            counter_uh = 0
            counter_um = 0
            timer_pauses = 0
            timer_uh = 0
            timer_um = 0

            # Perform the feature extraction of pauses on the clean data
            for tok in json_text["tokens"]:
                Text = tok["text"].lower()
                if (Text == "sil"):
                    pause_time = tok["end_time"] - tok["start_time"]
                    timer_pauses += pause_time
                    counter_pauses += 1
                elif (Text == "uh"):
                    uh_time = tok["end_time"] - tok["start_time"]
                    timer_uh += uh_time
                    counter_uh += 1
                elif (Text == "um"):
                    um_time = tok["end_time"] - tok["start_time"]
                    timer_um += um_time
                    counter_um += 1

            # Append values
            new_dict["pause_duration"].append(timer_pauses / elapsed_time)
            new_dict["no_of_pauses"].append(counter_pauses / elapsed_time)

            new_dict["uh_duration"].append(timer_uh / elapsed_time)
            new_dict["no_of_uh"].append(counter_uh / elapsed_time)

            new_dict["um_duration"].append(timer_um / elapsed_time)
            new_dict["no_of_um"].append(counter_um / elapsed_time)

print(len(new_dict["id"]), "Gweek files getting checked")
print("create dataframe from dictionary")
df = pd.DataFrame.from_dict(new_dict)

# Normalizing data using function from
# https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
print("Normalizing data")

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name != "id":
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if max_value * min_value != 0:
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            else:
                result[feature_name] = df[feature_name]
    return result


df = normalize(df)

print("Preprocessing done")

# Apply SVD to the data before passing it to the model
X = df.iloc[:, 1:]

svd = TruncatedSVD(n_components=3)
svd.fit(X)
X = svd.transform(X)

prediction = gnb.predict(X)



# Give the estimate of how many recordings are read

df_est = pd.DataFrame({'id': df["id"],
                       'nb': prediction})


avg_nb = sum(df_est["nb"]) / len(df_est["nb"])
print("Naive Bayes predicts {0:.1f}% read recordings".format(100 - avg_nb * 100))

# List of id's and labels
list_of_csv = []
for i in range(len(df_est["nb"])):
    val = df_est["nb"][i]
    if val == 0:
        list_of_csv.append([df_est["id"][i], "read"])
    else:
        list_of_csv.append([df_est["id"][i], "planned"])

# Create CSV file with all labels, save it in same folder as code

import csv

with open('folder_id_label.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(list_of_csv)

csvFile.close()