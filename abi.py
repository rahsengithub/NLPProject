import os, json
import pandas as pd
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS


import matplotlib.pyplot as plt


# this finds our json files
path_to_json = 'data_folder'
stopwords = set(STOPWORDS)
stop_list = list(stopwords) + ["i", "you", "me", "the", "it", "hi", "my", "am", "he", "she", "we", "a"]
#stop_list = ["i", "you", "me", "the", "it", "hi", "my", "am", "he", "she", "we", "a"]
stop_list = set(stop_list)
VOCAB = set()
vocab_list = []
score_dict = {}
tot = {}
voc_dict = {}


new_dict = {}
new_dict["f_1"] = []
new_dict["f_2"] = []
new_dict["f_3"] = []
new_dict["label"] = []

setter = set()
inter = set()
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
            if score > 95: new_dict["label"].append(1)
            else: new_dict["label"].append(0)
    
    
            doc_vocab = set()
            counter = 0
            for tok in json_text["tokens"]:
                Text = tok["text"].lower()
                counter += 1
                doc_vocab.add(Text)
                VOCAB.add(Text)
                vocab_list.append(Text)

    
            for stop_word in stop_list:
                try:
                    doc_vocab.remove(stop_word)
                    VOCAB.remove(stop_word)
                except: continue
    
            # new words pr min
            f_1 = len(doc_vocab)/json_text["elapsed_time"]
            feature_dict["new_words_pr_min"] = f_1
            
    
            # repeated words pr min
            f_2 = (counter - len(doc_vocab))/json_text["elapsed_time"]
            feature_dict["repeated_words_pr_min"] = f_2
            
            new_dict["f_1"].append(f_1)
            new_dict["f_2"].append(f_2)
    
            
            feature_dict["time"] = json_text["elapsed_time"]
            tot[id_] = feature_dict
    
            voc_dict[id_] = doc_vocab
    
            union = setter.union(doc_vocab-inter)
            intersect = setter.intersection(doc_vocab)
            setter = union - intersect
            inter = intersect
            
            if (f_1 == 0) & (f_2 == 0):
                print('ERRORRRRRR')
                print(subdir)

for id in tot.keys():
    voc = voc_dict[id]
    time = tot[id]["time"]
    f_3 = len(voc.intersection(setter))/(time)
    new_dict["f_3"].append(f_3)



print("Preprocessing done")
print("Preparing classification task")



# create dataframe
#import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm  

print("create dataframe from dictionary")
df = pd.DataFrame.from_dict(new_dict)
#df.drop("154", axis = 0)
print(df)


y = df.iloc[:,-1]
X = df.iloc[:,:-1]


#SVM = svm.LinearSVC()
#SVM.fit(X, y)
#SVM.predict(X.iloc)
#round(SVM.score(X,y), 4)


LR = LogisticRegression(random_state=0,solver='lbfgs', multi_class='ovr').fit(X, y)
LR.predict(X)
print(round(LR.score(X,y), 4))


def generate_wordcloud(dictionary):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_list,
        max_words=200,
        max_font_size=80,
        random_state=42
    ).generate_from_frequencies(dictionary)

    fig = plt.figure(1)
    plt.imshow(wordcloud)
    #plt.title(year_name)
    plt.axis('off')
    plt.show()
    #fig.savefig(year_name + ".png", dpi=1000)


# for x in year_dictionary.keys():
#     print("Generating wordCloud for year: " + str(x))
#     generate_wordcloud(year_dictionary[x], x)
# print("completed")

generate_wordcloud(vocab_list)