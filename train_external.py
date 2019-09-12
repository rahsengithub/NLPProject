
# Libraries
import os, json
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

# specify path
path_to_json = '../data_folder/ted_audio_api'

# Convert to python language
null = None
false = False
true = True
NoneType = None

new_dict = {}
new_dict["pause_duration"] = []
new_dict["no_of_pauses"] = []
new_dict["uh_duration"] = []
new_dict["no_of_uh"] = []
new_dict["um_duration"] = []
new_dict["no_of_um"] = []
new_dict["binary_classifier"] = []

pause_duration = []
pause_per_sec = []
setter = set()
inter = set()

filled_ted = 0
filled_audio = 0
print("Preprocessing external data...")
# loop through all external files from API
for subdir, dirs, files in os.walk(path_to_json):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(subdir, file)

            with open(path) as f:
                json_text = json.loads(f.read())

            # Extract tokens and score
            json_text = eval(json_text)
            score = json_text["Score"]
            json_text = json_text["AnalysisResult"]

            feature_dict = {}

            # Drop outliers
            if score < 5:
                print("Outlier, score less than 5")
                continue

            if not json_text["tokens"]:
                print("Outlier, no tokens")
                continue

            elapsed_time = json_text["performance"]["elapsed_time"]
            counter_pauses = 0
            counter_uh = 0
            counter_um = 0
            timer_pauses = 0
            timer_uh = 0
            timer_um = 0

            # Loop through each list of tokens to extract pause information
            # Including silent pause and filled pause
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
                    if "TED" in file:
                        filled_ted += 1
                    else:
                        filled_audio += 1
                elif (Text == "um"):
                    um_time = tok["end_time"] - tok["start_time"]
                    timer_um += um_time
                    counter_um += 1
                    if "TED" in file:
                        filled_ted += 1
                    else:
                        filled_audio += 1

            new_dict["pause_duration"].append(timer_pauses / elapsed_time)
            new_dict["no_of_pauses"].append(counter_pauses / elapsed_time)

            new_dict["uh_duration"].append(timer_uh / elapsed_time)
            new_dict["no_of_uh"].append(counter_uh / elapsed_time)

            new_dict["um_duration"].append(timer_um / elapsed_time)
            new_dict["no_of_um"].append(counter_um / elapsed_time)

        # adding Read and planned label
        if "TED" in file:
            binary = 1
        else:
            binary = 0
        new_dict["binary_classifier"].append(binary)

print(sum(new_dict["binary_classifier"]))

# Create df
print("Create dataframe from dictionary")
df = pd.DataFrame.from_dict(new_dict)

# Normalizing data frame with function collected from
# https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
print("Normalizing data")
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name != "binary_classifier":
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if max_value * min_value != 0:
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            else:
                result[feature_name] = df[feature_name]
    return result


df = normalize(df)

print("Preprocessing done")
print("Preparing classification task")

print("---------------------------")
print("Data Exploration")


# Some Data exploration including one correlation plot
tot_data = len(df["pause_duration"])

print("Correlation plot")
plt.plot(df["pause_duration"], df["no_of_pauses"], 'bo')
plt.xlabel('Pause Duration')
plt.ylabel('Pause Frequency')
plt.title('Correlation plot, {0} datapoints'.format(tot_data))
plt.show()

print("Total number of filled pauses for {0} TED talks: {1}".format(tot_data, filled_ted))
print("Total number of filled pauses for {0} Audio books: {1}".format(tot_data, filled_audio))
print("Prcentage filled pauses in TED: {0:.3f}".format(filled_ted / (filled_ted + filled_audio)))

# Training
print("Training on pause features")
print("--------------------------------------")
print("Preparing binary classification task")


from sklearn.decomposition import TruncatedSVD


# Training starts here to classify read/planned

y = df.iloc[:, -1]
X = df.iloc[:, :-1]


# Run 4 algorithms once splitted data
def algo(a_train, a_test, b_train, b_test):
    ############Logistic Regression #####################

    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(a_train, b_train)
    predicted = LR.predict(a_test)
    print("Accuracy, logistic regression: {0:.3f}".format(LR.score(a_test, b_test)))
    print("F1 score, logistic regression: {0:.3f}".format(f1_score(predicted, b_test)))

    ############Random Forest classifier#####################

    rf = RandomForestClassifier()
    rf.fit(a_train, b_train)
    predictions = rf.predict(a_test)

    print("Accuracy for RandomForestClassifier: {0:.3f}".format(metrics.accuracy_score(b_test, predictions)))
    print("F1 score, RandomForestClassifier: {0:.3f}".format(f1_score(predictions, b_test)))

    # ######################## Naive Bayes classifier###############

    gnb = GaussianNB()
    gnb.fit(a_train, b_train)
    prediction = gnb.predict(a_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy for Naive Bayes Classifier: {0:.3f}".format(metrics.accuracy_score(b_test, prediction)))
    print("F1 score, Naive Bayes: {0:.3f}".format(f1_score(prediction, b_test)))

    ########################## Gradient Classifier###################

    gb = GradientBoostingClassifier()
    gb.fit(a_train, b_train)
    p = gb.predict(a_test)
    print("Accuracy for Gradient Boosting: {0:.3f}".format(gb.score(a_test, b_test)))
    print("F1 score, Gradient Boosting: {0:.3f}".format(f1_score(predictions, b_test)))

    return LR, rf, gnb, gb


# Split to train and test and do train the binary classification model
a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.33, random_state=42)
LR, rf, gnb, gb = algo(a_train, a_test, b_train, b_test)

# Split and train after SVD is applied and has reduced dimensions from 6 to 3
print("------------------------------")
print("Run SVD on features to deal with sparsity")
svd = TruncatedSVD(n_components=3)
svd.fit(X)
X = svd.transform(X)

print("Train on new features")
a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.33, random_state=42)
LR, rf, gnb, gb = algo(a_train, a_test, b_train, b_test)


# Dump the Naive Bayes model as a .joblib file

from joblib import dump, load
dump(gnb, 'naive_bayes_model.joblib')
