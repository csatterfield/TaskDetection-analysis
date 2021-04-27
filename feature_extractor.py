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
    total = sum(scores.values())

    if(total != 0):
        for score in scores:
            scores[score] = scores[score]/total

    m1 =  max(scores, key=scores.get)
    s1 = scores[m1]
    del(scores[m1])
    m2 = max(scores, key=scores.get)
    s2 = scores[m2]
    return {"labels": [m1, m2], "scores": [s1, s2]}

def predict_rake(tasks, phrase):
    search_terms = word_tokenize(phrase)
    search_terms = [x for x in search_terms if not x in stop_words]
    scores = dict()

    r = Rake()
    for task in [1,2,3,4,5,6]:
        words = []
        for snapshot in tasks[task]:
            r.extract_keywords_from_text(snapshot)
            tokens = r.get_ranked_phrases()
            snapshot_words = [x for x in tokens if len(x) > 2]
            snapshot_words = [x for x in snapshot_words if x in model]
            words.extend(snapshot_words)

        c = Counter(words)
        occurs = 0
        for word in search_terms:
            if word in c:
                occurs += c[word]
        
        scores[task] = occurs


    return get_prediction(scores)



def predict_simple(tasks, phrase):
    search_terms = word_tokenize(phrase)
    search_terms = [x for x in search_terms if not x in stop_words]
    scores = dict()

    for task in [1,2,3,4,5,6]:
        words = []
        for snapshot in tasks[task]:
            tokens = word_tokenize(snapshot)
            words.extend(tokens)

        c = Counter(words)
        occurs = 0
        for word in search_terms:
            if word in c:
                occurs += c[word]
        
        scores[task] = occurs
    
    return get_prediction(scores)



def predict_word2vec(tasks, phrase):
    scores = dict()

    for task in [1,2,3,4,5,6]:
        words = []
        for snapshot in tasks[task]:
            tokens = word_tokenize(snapshot)
            snapshot_words = [x for x in tokens if not x in stop_words]
            snapshot_words = [x for x in snapshot_words if len(x) > 2]
            snapshot_words = [x for x in snapshot_words if x in model]
            words.extend(snapshot_words)

        c = Counter(words)
        #c = {k:floor(log(v)) for (k,v) in c.items() if v >= 5 }
        
        v1 = np.zeros(300)
        for word in c:
            v1 += model[word] * c[word]
        
        v2 = np.zeros(300)
        search_terms = word_tokenize(phrase)
        search_terms = [x.lower() for x in search_terms]
        search_terms = [x for x in search_terms if not x in stop_words]
        search_terms = [x for x in search_terms if x in model]

        for word in search_terms:
            v2 += model[word]

        scores[task] = sim(v1,v2)
    return get_prediction(scores)

def predict_word2vec_keywords(tasks, phrase):
    r = Rake()
    scores = dict()

    for task in [1,2,3,4,5,6]:
        words = []
        for snapshot in tasks[task]:
            r.extract_keywords_from_text(snapshot)
            tokens = r.get_ranked_phrases()
            snapshot_words = [x for x in tokens if len(x) > 2]
            snapshot_words = [x for x in snapshot_words if x in model]
            words.extend(snapshot_words)

        c = Counter(words)
        #c = {k:floor(log(v)) for (k,v) in c.items() if v >= 5 }
        
        v1 = np.zeros(300)
        for word in c:
            v1 += model[word] * c[word]
        
        v2 = np.zeros(300)
        search_terms = word_tokenize(phrase)
        search_terms = [x.lower() for x in search_terms]
        search_terms = [x for x in search_terms if not x in stop_words]
        search_terms = [x for x in search_terms if x in model]
        for word in search_terms:
            v2 += model[word]

        scores[task] = sim(v1,v2)
    return get_prediction(scores)

def sim(v1, v2):
    return cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-2))[0][0]

def predict_all(phrase):
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
        sample = {"label": phrase["expected"], "participant": participant}
        #### Screen Captures
        #TF
        tasks = te_sc.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_simple(tasks, phrase["phrase"])
        sample["sc_tf_label1"] = predicted["labels"][0]
        sample["sc_tf_score1"] = predicted["scores"][0]
        sample["sc_tf_label2"] = predicted["labels"][1]
        sample["sc_tf_score2"] = predicted["scores"][1]

        #RAKE
        tasks = te_sc.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_rake(tasks, phrase["phrase"])
        sample["sc_rake_label1"] = predicted["labels"][0]
        sample["sc_rake_score1"] = predicted["scores"][0]
        sample["sc_rake_label2"] = predicted["labels"][1]
        sample["sc_rake_score2"] = predicted["scores"][1]
        
        #W2V
        tasks = te_sc.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_word2vec(tasks, phrase["phrase"])
        sample["sc_w2v_label1"] = predicted["labels"][0]
        sample["sc_w2v_score1"] = predicted["scores"][0]
        sample["sc_w2v_label2"] = predicted["labels"][1]
        sample["sc_w2v_score2"] = predicted["scores"][1]
        
        ### Window Titles
        #TF
        tasks = te_wt.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_simple(tasks, phrase["phrase"])
        sample["wt_tf_label1"] = predicted["labels"][0]
        sample["wt_tf_score1"] = predicted["scores"][0]
        sample["wt_tf_label2"] = predicted["labels"][1]
        sample["wt_tf_score2"] = predicted["scores"][1]

        #RAKE
        tasks = te_wt.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_rake(tasks, phrase["phrase"])
        sample["wt_rake_label1"] = predicted["labels"][0]
        sample["wt_rake_score1"] = predicted["scores"][0]
        sample["wt_rake_label2"] = predicted["labels"][1]
        sample["wt_rake_score2"] = predicted["scores"][1]
        
        #W2V
        tasks = te_wt.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_word2vec(tasks, phrase["phrase"])
        sample["wt_w2v_label1"] = predicted["labels"][0]
        sample["wt_w2v_score1"] = predicted["scores"][0]
        sample["wt_w2v_label2"] = predicted["labels"][1]
        sample["wt_w2v_score2"] = predicted["scores"][1]

        results.append(sample)


    return results

    

if __name__ == "__main__":
    phrases = [{"phrase": "Todo list app: determine best from reviews", "expected":4, "author": "thomas"},
    {"phrase": "research on deep learning: layer weights, history and recent change, GPU vs CPU", "expected": 5, "author": "thomas"},
    {"phrase": "identifying duplicate bugs in project", "expected": 1, "author": "thomas"},
    {"phrase": "compare popular Todo list app (similarities and differences)", "expected": 3, "author": "thomas"},
    {"phrase": "visualizing app's impact on interruptions and workday", "expected": 2, "author": "thomas"},
    {"phrase": "research on blockchain: distributed ledger & proof-of-work", "expected": 6, "author": "thomas"},
    {"phrase": "Identify useful app reviews for to-do list apps: Microsoft To-do, Wunderlist and Todoist.", "expected": 4, "author": "gail"},
    {"phrase": "Prepare for presentation on deep learning to CTO", "expected": 5, "author": "gail"},
    {"phrase": "Find whether there are duplicates for bugs 2264, 2268, 2271 and 2777", "expected": 1, "author": "gail"},
    {"phrase": "Do market research on the to-do apps: Microsoft To-Do, Wunderlist and Todoist", "expected": 3, "author": "gail"},
    {"phrase": "Find a software library to visualize line charts, pie charts, etc.", "expected": 2, "author": "gail"},
    {"phrase": "Prepare answers for follow-up questions on Blockchain about distributed ledgers and proof-of-work.", "expected": 6, "author": "gail"}]
    
    final_results = []

    for phrase in phrases:
        print("START ----- ")
        print("Searching for phrase: " + phrase["phrase"])
        final_results.extend(predict_all(phrase))
        print("END -----")
    
    df = pd.DataFrame(final_results)
    df.to_excel("features_both.xlsx")