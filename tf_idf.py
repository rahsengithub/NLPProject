import os, json
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
# this finds our json files
path_to_json = '../data_folder'

stop_list = ["i", "you", "me", "the", "it", "hi", "my", "am", "he", "she", "we", "a"]

score_list = []
label_list = []
data = []

print("pre-processing files.....")
# we need both the json and an index number so use enumerate()
for subdir, dirs, files in os.walk(path_to_json):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(subdir, file)

            with open(path, 'r') as f:
                json_text = json.load(f)
            id_ = json_text["id"]
            feature_dict = {}
            score = json_text["score"]
            score_list.append(score)
            if score > 95:
                label_list.append(1)
            else:
                label_list.append(0)

            text = ""
            for tok in json_text["tokens"]:
                Text = tok["text"].lower()
                if Text not in stop_list:
                    text += " " + Text

            data.append(text)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
transformer = TfidfTransformer()
data = transformer.fit_transform(X).toarray()

# splitting
from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(data, np.asarray(label_list), test_size=0.33, random_state=42)


print("Preprocessing done")

print("--------------------------------------")
print("Preparing classification task")

# create dataframe
# import sklearn as sk
from sklearn.linear_model import LogisticRegression


LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(a_train, b_train)
predicted = LR.predict(a_test)
y_true = np.asarray(label_list)
print("F1-score, TF-IDF")
print(f1_score(predicted, b_test, average='weighted'))
#print("Accuracy, TF-IDF")
#print(round(LR.score(predicted, b_test), 4))


print("Preparing regression task")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
a_train, a_test, b_train, b_test = train_test_split(data, np.asarray(score_list), test_size=0.33, random_state=42)


LinR = LinearRegression().fit(a_train, b_train)
predicted = LinR.predict(a_test)
rmse = sqrt(mean_squared_error(predicted, b_test))
print("Root Mean square error")
print(rmse)

print(data.shape)
