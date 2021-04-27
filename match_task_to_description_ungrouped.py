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
import fasttext
import fasttext.util
import pickle
import os


#model = KeyedVectors.load("models/normalized.model")
fasttext.util.download_model('en', if_exists='ignore')
model = fasttext.load_model('cc.en.300.bin')
stop_words = set(stopwords.words('english'))
vocab = set(model.words)

cache={}
try:
    with open("cache.pkl", "rb") as f:
        cache=pickle.load(f)
except:
    pass

def shuffleDict(dictionary):
    keys = list(dictionary.keys())
    random.shuffle(keys)
    shuffled = {}
    for key in keys:
        shuffled[key] = dictionary[key]
    
    return shuffled

def equals(prediction, expected):
    if(prediction == expected):
        return 1
    else:
        return 0

def get_prediction(scores):
    scores = shuffleDict(scores)
    return max(scores, key=scores.get)

def normalize_score(scores):
    norm = np.linalg.norm(list(scores.values()))
    for score in scores:
        scores[score] = scores[score]/norm
    return scores

def predict_rake(tasks, order, phrases):
    predictions = []
    expected = []
    durations = []
    r = Rake()
    for task, actual in zip(tasks, order):
        scores = dict()
        cover_scores = dict()
        expected.append(actual)
        words = []
        cover = {}
        
        for _, row in phrases.iterrows():
            search_terms = word_tokenize(row["phrase"])
            search_terms = [x for x in search_terms if not x in stop_words]
            search_terms = [x for x in search_terms if len(x) > 2]
            search_terms = [x for x in search_terms if x in model]

            occurs = 0
            coverage = 0
            covered = []
            
            for word in search_terms:
                if word in task:
                    occurs += task[word]
                    coverage += 1
                    covered.append(word)
                coverage = coverage/len(search_terms)
            
            scores[row["expected"]] = occurs
            cover_scores[row["expected"]] = coverage

            cover[row["expected"]] = [(x, task[x]) for x in covered]

        scores = normalize_score(scores)
        cover_scores = normalize_score(scores)

        for key in scores.keys():
            scores[key] = scores[key] * 1 + cover_scores[key] * 0

        predictions.append(get_prediction(scores))

    return predictions, expected



def predict_simple(tasks, order, phrases):
    predictions = []
    expected = []
    durations = []
    sim_vocab = []

    for task, actual in zip(tasks, order):
        scores = dict()
        cover_scores = dict()
        expected.append(actual)
        words = []
        cover = {}
        
        for _, row in phrases.iterrows():
            search_terms = word_tokenize(row["phrase"])
            search_terms = [x for x in search_terms if not x in stop_words]
            search_terms = [x for x in search_terms if len(x) > 2]
            search_terms = [x for x in search_terms if x in vocab]

            occurs = 0
            coverage = 0
            covered = []
            
            for word in search_terms:
                if word in task:
                    occurs += task[word]
                    coverage += 1
                    covered.append(word)
                else:
                    v1 = model.get_word_vector(word)
                    for w2 in task:
                        if((word, w2) not in cache):
                            v2 = model.get_word_vector(w2)
                            cache[(word, w2)] = sim(v1, v2)
                        if(cache[(word, w2)] > 0.8):
                            occurs += task[w2]
                            sim_vocab.append(w2)
                        
                coverage = coverage/len(search_terms)
            
            scores[row["expected"]] = occurs
            cover_scores[row["expected"]] = coverage

            cover[row["expected"]] = [(x, task[x]) for x in covered]

        scores = normalize_score(scores)
        cover_scores = normalize_score(scores)

        for key in scores.keys():
            scores[key] = scores[key] * 1 + cover_scores[key] * 0

        predictions.append(get_prediction(scores))

    return predictions, expected
    
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

    predictions = []
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

        predictions.append(get_prediction(scores))
        durations.append(task["duration"])
    return predictions, expected, durations

def sim(v1, v2):
    return cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0]

def create_results(method, predicted, expected, author, participant):
    return [{"method": method, "predicted": x, "expected": y, "author": author, "participant": participant, "correct": equals(x, y)} for x,y in zip(predicted, expected)]

def predict_all(task_descriptions, author):
    path_to_data = "../archives"
    results = []
    #all participants
    #participants = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]
    #reduced participant set for window titles - some participants are missing window title data
    participants = ["P01", "P02", "P03", "P04", "P05", "P06", "P14", "P15", "P16", "P17", "P18", "P19"]

    task_extractor = te.ScreenshotTaskExtractor(set(model.words))
    
    
    print("SIMPLE METHOD")
    for participant in participants:
        print(participant)
        tasks, order = task_extractor.get_tasks_for_participant(path_to_data, participant, using_widf=True, ungrouped=True)
        predicted,expected = predict_simple(tasks, order, task_descriptions)
        results.extend(create_results("simple", predicted, expected, author, participant))
    '''
    def rake(snapshot):
        r = Rake()
        r.extract_keywords_from_text(snapshot)
        return r.get_ranked_phrases()

    print("RAKE METHOD")
    for participant in participants:
        print(participant)
        tasks, order = task_extractor.get_tasks_for_participant(path_to_data, participant, using_widf=True, ungrouped=True, filter=rake)
        predicted,expected = predict_rake(tasks, order, task_descriptions)
        results.extend(create_results("rake", predicted, expected, author, participant))

    print("WORD2VEC METHOD")
    for participant in participants:
        print(participant)
        tasks = task_extractor.get_tasks_for_participant(path_to_data, participant, ungrouped=True)
        predicted,expected,durations = predict_word2vec(tasks, task_descriptions)
        results.extend(create_results("w2v", predicted, expected, durations, author, participant))
    print("WORD2VEC KEYWORDED METHOD")
    for participant in participants:
        print(participant)
        tasks = task_extractor.get_tasks_for_participant(path_to_data, participant, ungrouped=True)
        predicted,expected,durations = predict_word2vec_keywords(tasks, task_descriptions)
        results.extend(create_results("w2v_rake", predicted, expected, durations, author, participant))
    '''

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
    df.to_excel("matching_results_upgrouped.xlsx")

    with open("cache.pkl", "wb") as f:
        pickle.dump(cache, f)