import task_extractor as te
from rake_nltk import Rake
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from math import log, floor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import textrank
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


model = KeyedVectors.load("models/normalized.model")
stop_words = set(stopwords.words('english'))

def equals(prediction, expected):
    if(prediction == expected):
        return 1
    else:
        return 0

def get_prediction(scores):
    return max(scores, key=scores.get)

def get_confidence(scores):
    total = sum(scores.values())

    if(total != 0):
        for score in scores:
            scores[score] = scores[score]/total

    return scores

def predict_rake(tasks, phrases):
    confidences = []
    expected = []
    durations = []
    r = Rake()
    for task in tasks:
        scores = dict()
        expected.append(task["task"])
        words = []

        for snapshot in task["snapshots"]:
            r.extract_keywords_from_text(snapshot)
            tokens = r.get_ranked_phrases()
            snapshot_words = [x for x in tokens if len(x) > 2]
            snapshot_words = [x for x in snapshot_words if x in model]
            words.extend(snapshot_words)

        c = Counter(words)
        for _, row in phrases.iterrows():
            search_terms = word_tokenize(row["phrase"])
            search_terms = [x for x in search_terms if not x in stop_words]
            occurs = 0
            for word in search_terms:
                if word in c:
                    occurs += c[word]
            
            scores[row["expected"]] = occurs/len(tasks)
        
        confidences.append(get_confidence(scores))
        durations.append(task["duration"])
    return confidences, expected



def predict_simple(tasks, phrases):
    confidences = []
    expected = []
    durations = []
    for task in tasks:
        scores = dict()
        expected.append(task["task"])
        words = []
        for snapshot in task["snapshots"]:
            tokens = word_tokenize(snapshot)
            words.extend(tokens)

        c = Counter(words)
        for _, row in phrases.iterrows():
            search_terms = word_tokenize(row["phrase"])
            search_terms = [x for x in search_terms if not x in stop_words]
            occurs = 0
            for word in search_terms:
                if word in c:
                    occurs += c[word]
            
            scores[row["expected"]] = occurs/len(tasks)

        confidences.append(get_confidence(scores))

    return confidences, expected

def apply_tf_idf_filter(tasks, words):
    task_words = []
    idf_counter = Counter()
    for task in tasks:
        tmp = []
        for snapshot in task["snapshots"]:
            tokens = word_tokenize(snapshot)
            snapshot_words = [x for x in tokens if not x in stop_words]
            snapshot_words = [x for x in snapshot_words if len(x) > 2]
            snapshot_words = [x for x in snapshot_words if x in model]
            tmp.extend(snapshot_words)
        idf_counter += Counter(set(tmp))

    words = Counter(words)
    word_scores = {}
    for word in words:
        tf = words[word] / sum(words.values())
        idf = idf_counter[word]/len(tasks)
        word_scores[word] = tf * idf
    word_scores_top = Counter(word_scores).most_common(50)
    return Counter({x[0]: x[1] for x in word_scores_top})


def predict_word2vec(tasks, phrases):

    confidences = []
    expected = []
    durations = []
    for task in tasks:
        scores = dict()
        expected.append(task["task"])
        words = []
        for snapshot in task["snapshots"]:
            tokens = word_tokenize(snapshot)
            snapshot_words = [x for x in tokens if not x in stop_words]
            snapshot_words = [x for x in snapshot_words if len(x) > 2]
            snapshot_words = [x for x in snapshot_words if x in model]
            words.extend(snapshot_words)

        c = apply_tf_idf_filter(tasks, words)
        #c = {k:floor(log(v)) for (k,v) in c.items() if v >= 5 }
        
        v1 = np.zeros(300)
        for word in c:
            v1 += model[word] * c[word]

        for _, row in phrases.iterrows():
            
            v2 = np.zeros(300)
            search_terms = word_tokenize(row["phrase"])
            search_terms = [x.lower() for x in search_terms]
            search_terms = [x for x in search_terms if not x in stop_words]
            search_terms = [x for x in search_terms if len(x) > 2]
            search_terms = [x for x in search_terms if x in model]

            for word in search_terms:
                v2 += model[word]

            scores[row["expected"]] = sim(v1,v2)

        confidences.append(get_confidence(scores))
    return confidences, expected

def sim(v1, v2):
    return cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-2))[0][0]

def create_results(confidences, expected, participant, author):
    results = [{"participant": participant, "author": author, "expected": y, **x} for x, y in zip(confidences, expected)]
    print(results)
    return results

def predict_all(task_descriptions, author):
    path_to_data = "../archives"
    results = []
    #all participants
    #participants = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]
    #reduced participant set for window titles - some participants are missing window title data
    participants = ["P01", "P02", "P03", "P04", "P05", "P06", "P14", "P15", "P16", "P17", "P18", "P19"]
    te_sc = te.ScreenshotTaskExtractor()
    te_wt = te.WindowTitleTaskExtractor()

    for participant in participants:
        print(participant)
        #### Screen Captures
        tasks = te_sc.get_tasks_for_participant(path_to_data, participant, ungrouped=True)

        #RAKE
        confidences_rake, _ = predict_rake(tasks, task_descriptions)
        confidences_rake = [{"sc_rake_" + str(x): y[x] for x in y} for y in confidences_rake]
        #W2V
        confidences_w2v, _ = predict_word2vec(tasks, task_descriptions)
        confidences_w2v = [{"sc_w2v_" + str(x): y[x] for x in y} for y in confidences_w2v]

        confidences_sc = [{**y,**z} for y,z in zip(confidences_rake, confidences_w2v)]
        
        ### Window Titles
        tasks = te_wt.get_tasks_for_participant(path_to_data, participant, ungrouped=True)

        #TF
        confidences_tf, expected = predict_simple(tasks, task_descriptions)
        confidences_tf = [{"wt_tf_" + str(x): y[x] for x in y} for y in confidences_tf]
        #W2V
        confidences_w2v, _ = predict_word2vec(tasks, task_descriptions)
        confidences_w2v = [{"wt_w2v_" + str(x): y[x] for x in y} for y in confidences_w2v]

        confidences_wt = [{**x,**z} for x,z in zip(confidences_tf, confidences_w2v)]

        confidences = [{**x, **y} for x,y in zip(confidences_sc, confidences_wt)]
        results.extend(create_results(confidences, expected, participant, author))

    return results

    

if __name__ == "__main__":

    df = pd.read_excel("phrases.xlsx")    
    final_results = []

    for author in df["author"].unique():
        task_descriptions = df[df["author"] == author]
        print("START ----- ")
        print("Matching phrases by author: " + author)
        final_results.extend(predict_all(task_descriptions, author))
        print("END -----")
    
    df = pd.DataFrame(final_results)
    df.to_excel("confidences.xlsx")