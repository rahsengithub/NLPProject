
# Import necessary libraries

import os, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

# Data Cleaning + Feature Extraction

# Specify path
path_to_json = '../data_folder/json_train/'

# Words not to be a part of our vocabulary
stop_list = list(STOPWORDS) + ["sil", "uh", "um"]


VOCAB = set()
score_dict = {}
tot = {}
voc_dict = {}
vocab_list = []

new_dict = {}
new_dict["f_1"] = []
new_dict["f_2"] = []
new_dict["f_3"] = []
new_dict["scores"] = []
new_dict["binary_label"] = []
new_dict["multiclass_label"] = []

data = []
print("Extraction features....")
setter = set()
inter = set()

# Loop through all folders with generated json-files
for subdir, dirs, files in os.walk(path_to_json):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(subdir, file)
            with open(path, 'r') as f:
                json_text = json.load(f)
            id_ = file
            score = json_text["score"]

            # Remove Outliers

            # Don't include JSONs that are processed already
            if id_ in tot.keys():
                continue

            # Drop JSONs with score < 5
            if score < 5:
                print("Outlier, Score less than 5")
                continue


            # Drop JSONs without tokens
            if not json_text["tokens"]:
                print("Outlier, no tokens")
                continue

            # Loop through all tokens to add to vocabulary
            text = ""
            doc_vocab = set()
            counter = 0
            for tok in json_text["tokens"]:
                Text = tok["text"].lower()
                if Text not in stop_list:
                    text += " " + Text
                    counter += 1
                    doc_vocab.add(Text)
                    VOCAB.add(Text)
                    vocab_list.append(Text)

            data.append(text)
            new_dict["scores"].append(score)


            # feature1: new words per sec
            f_1 = len(doc_vocab) / json_text["elapsed_time"]
            if (f_1 < 0.15):
                print('ERRORRRRRR')
                print(subdir)

            feature_dict = {}
            feature_dict["new_words_pr_min"] = f_1

            # feature2: fraction of unique words for each user
            f_2 = len(doc_vocab) / (counter * json_text["elapsed_time"])
            feature_dict["repeated_words_pr_min"] = f_2

            new_dict["f_1"].append(f_1)
            new_dict["f_2"].append(f_2)

            feature_dict["time"] = json_text["elapsed_time"]
            tot[id_] = feature_dict

            voc_dict[id_] = doc_vocab

            # feature3: special words, given by one user only
            # build the vocabulary of special words (setter)
            union = setter.union(doc_vocab - inter)
            intersect = setter.intersection(doc_vocab)
            setter = union - intersect
            inter = intersect

            # Labelling process
            # Goal: set thresholds to get uniform distribution
            # Thresholds are set by looking at subset of 490 data points
            if score > 91:
                new_dict["binary_label"].append(1)
                if score > 95.5:
                    new_dict["multiclass_label"].append(3)
                else:
                    new_dict["multiclass_label"].append(2)
            else:
                new_dict["binary_label"].append(0)
                if (score < 91) & (score > 79):
                    new_dict["multiclass_label"].append(1)
                else:
                    new_dict["multiclass_label"].append(0)

# Complete feature3: look up in the setter vocabulary for each user
for id_ in tot.keys():
    voc = voc_dict[id_]
    time = tot[id_]["time"]
    f_3 = len(voc.intersection(setter)) / (time)
    new_dict["f_3"].append(f_3)



# Create df
print("training on",len(tot.keys()),"data points")
print("Create dataframe from dictionary")
df = pd.DataFrame.from_dict(new_dict)



# Before traning, get TFIDF representation of all JSON tokens
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
transformer = TfidfTransformer()
data = transformer.fit_transform(X).toarray()


# Normalizing. Function taken from the internet.
# https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
print("Normalizing data")
def normalizer(df):
    result = df.copy()
    for feature_name in df.columns:
        if "f_" in feature_name:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


df = normalizer(df)


print("Preprocessing done")

# Training for 2 methods

print("--------------------------------------")
print("F1-F3")
print("Preparing binary classification task")

# Features and Binary column
y = df.iloc[:, -2]
X = df.iloc[:, :-3]
average = "binary"


# Run 4 algorithms once splitted data
def algo(a_train, a_test, b_train, b_test, pr_cl):

    ############Logistic Regression #####################


    LR = LogisticRegression(solver='lbfgs', multi_class='ovr').fit(a_train, b_train)
    predicted = LR.predict(a_test)


    ############Random Forest classifier#####################


    rf = RandomForestClassifier()
    rf.fit(a_train, b_train)
    predictions = rf.predict(a_test)


    # ######################## Naive Bayes classifier###############

    gnb = GaussianNB()
    gnb.fit(a_train, b_train)
    prediction = gnb.predict(a_test)


    # Model Accuracy, how often is the classifier correct?

    ########################## Gradient Classifier###################


    gb = GradientBoostingClassifier()
    gb.fit(a_train, b_train)

    # Compute Accuracy and F1 score if binary
    if pr_cl == "binary":
        print("Accuracy, logistic regression: {0:.3f}".format(LR.score(a_test, b_test)))
        print("F1 score, logistic regression: {0:.3f}".format(f1_score(predicted, b_test)))

        print("Accuracy for RandomForestClassifier: {0:.3f}".format(metrics.accuracy_score(b_test, predictions)))
        print("F1 score, RandomForestClassifier: {0:.3f}".format(f1_score(predictions, b_test)))

        print("Accuracy for Naive Bayes Classifier: {0:.3f}".format(metrics.accuracy_score(b_test, prediction)))
        print("F1 score, Naive Bayes: {0:.3f}".format(f1_score(prediction, b_test)))

        print("Accuracy for Gradient Boosting: {0:.3f}".format(gb.score(a_test, b_test)))
        print("F1 score, Gradient Boosting: {0:.3f}".format(f1_score(predictions, b_test)))


    # Compute Accuracy and F1 score if multiclass
    else:
        print("Accuracy, logistic regression: {0:.3f}".format(LR.score(a_test, b_test)))
        print("F1 score, logistic regression: {0:.3f}".format(f1_score(predicted, b_test, average='weighted')))

        print("Accuracy for RandomForestClassifier: {0:.3f}".format(metrics.accuracy_score(b_test, predictions)))
        print("F1 score, RandomForestClassifier: {0:.3f}".format(f1_score(predictions, b_test, average='weighted')))

        print("Accuracy for Naive Bayes Classifier: {0:.3f}".format(metrics.accuracy_score(b_test, prediction)))
        print("F1 score, Naive Bayes: {0:.3f}".format(f1_score(prediction, b_test, average='weighted')))

        print("Accuracy for Gradient Boosting: {0:.3f}".format(gb.score(a_test, b_test)))
        print("F1 score, Gradient Boosting: {0:.3f}".format(f1_score(predictions, b_test, average='weighted')))


# Split for binary and pass to function
a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.20, random_state=42)
algo(a_train, a_test, b_train, b_test, average)

print("--------------------------------------")
print("F1-F3")
print("Preparing MULTICLASS classification task")

# Split for multiclass and pass to function
y = df.iloc[:, -1]
average = "weighted"
a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.20, random_state=42)
algo(a_train, a_test, b_train, b_test, average)




# ONLY TF-IDF AS PREDICTIVE FEATURE
print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("TF-IDF without SVD")
print("Preparing BINARY classification task")

# Split for binary, TFIDF
y = df.iloc[:, -2]
average = "binary"
a_train, a_test, b_train, b_test = train_test_split(data, y, test_size=0.20, random_state=42)
algo(a_train, a_test, b_train, b_test, average)

##################################################################

print("--------------------------------------")
print("Preparing MULTICLASS classification task")

# Multiclass, TFIDF
y = df.iloc[:, -1]
average = "weighted"
a_train, a_test, b_train, b_test = train_test_split(data, y, test_size=0.20, random_state=42)
algo(a_train, a_test, b_train, b_test, average)


print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

# Run SVD on TFIDF matrix at is large and very sparse

print("Run SVD on Data frame to deal with sparsity")


raw_data = normalize(data, axis = 0)

svd = TruncatedSVD(n_components=5)
svd.fit(raw_data)
new_data = svd.transform(raw_data)

print("Preparing BINARY classification task, TF-IDF with SVD")
# Split and pass to function, binary
y = df.iloc[:, -2]
average = "binary"
a_train, a_test, b_train, b_test = train_test_split(new_data, y, test_size=0.20, random_state=42)
algo(a_train, a_test, b_train, b_test, average)


print("--------------------------------------")
print("Preparing MULTICLASS classification task, TF-IDF with SVD")

# SPlit and pass, multiclass
y = df.iloc[:, -1]
average = "weighted"
a_train, a_test, b_train, b_test = train_test_split(new_data, y, test_size=0.20, random_state=42)
algo(a_train, a_test, b_train, b_test, average)


# Compute RMSE from regression

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

################### Regression########################
print("--------------------------------------")
print("Preparing regression task")

def regressor(a_train, a_test, b_train, b_test):
    LinR = LinearRegression().fit(a_train, b_train)
    predicted = LinR.predict(a_test)
    rmse = sqrt(mean_squared_error(predicted, b_test))
    print("Root Mean square error")
    print(rmse)

# Score data
y = df.iloc[:, -3]

# Regression for each 3 methods
print("F1-F3")
a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.20, random_state=42)
regressor(a_train, a_test, b_train, b_test)



print("TF-IDF")
a_train, a_test, b_train, b_test = train_test_split(data, y, test_size=0.20, random_state=42)
regressor(a_train, a_test, b_train, b_test)


print("TF-IDF, SVD")
a_train, a_test, b_train, b_test = train_test_split(new_data, y, test_size=0.20, random_state=42)
regressor(a_train, a_test, b_train, b_test)

