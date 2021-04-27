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
        
        scores[task] = occurs/len(tasks[task])

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
        
        scores[task] = occurs/len(tasks[task])
    
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

def predict_random():
    return random.randint(1,6)

def sim(v1, v2):
    return cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-2))[0][0]

def create_result(method, predicted, expected, author, participant):
    return {"method": method, "predicted": predicted, "expected": expected, "author": author, "participant": participant, "correct": equals(predicted, expected)}

def predict_all(phrase):
    path_to_data = "../archives"
    results = []
    #all participants
    #participants = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]
    #reduced participant set for window titles - some participants are missing window title data
    participants = ["P01", "P02", "P03", "P04", "P05", "P06", "P14", "P15", "P16", "P17", "P18", "P19"]

    task_extractor = te.ScreenshotTaskExtractor()
    
    print("SIMPLE METHOD")
    for participant in participants:
        print(participant)
        tasks = task_extractor.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_simple(tasks, phrase["phrase"])
        results.append(create_result("simple", predicted, phrase["expected"], phrase["author"], participant))    
    print("RAKE METHOD")
    for participant in participants:
        print(participant)
        tasks = task_extractor.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_rake(tasks, phrase["phrase"])
        results.append(create_result("rake", predicted, phrase["expected"], phrase["author"], participant))
    
    print("WORD2VEC METHOD")
    for participant in participants:
        print(participant)
        tasks = task_extractor.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_word2vec(tasks, phrase["phrase"])
        results.append(create_result("w2v", predicted, phrase["expected"], phrase["author"], participant))

    print("WORD2VEC KEYWORDED METHOD")
    for participant in participants:
        print(participant)
        tasks = task_extractor.get_tasks_for_participant(path_to_data, participant)
        predicted = predict_word2vec_keywords(tasks, phrase["phrase"])
        results.append(create_result("w2v_rake", predicted, phrase["expected"], phrase["author"], participant))


    return results

    

if __name__ == "__main__":
    
    phrases = pd.read_excel("phrases.xlsx")    
    final_results = []

    for _, row in phrases.iterrows():
        print("START ----- ")
        print("Searching for phrase: " + row["phrase"])
        final_results.extend(predict_all(row))
        print("END -----")
    
    df = pd.DataFrame(final_results)
    df.to_excel("matching_results_tfidf.xlsx")