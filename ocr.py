# created by Chris Satterfield on 2017-07-21

from multiprocessing import Manager, Pool
from datetime import datetime
import subprocess
from subprocess import Popen, PIPE
import sys
import re
import os
import os.path
import textrank
import numpy as np
import copy
import pickle
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string



#load the dictionary
'''
with open('./dict/dict.txt') as f:
    d = f.read().splitlines()[44:]

    d = [x.lower() for x in d]
    d = [re.sub(r'[^a-zA-Z]+', '', x) for x in d]

    dictionary = set(d)
'''

def clean(task):

	newTask = task.replace('\n', ' ') #remove all new lines

	remove = string.punctuation
	remove = remove.replace("-", "")
	pattern = r"[{}]".format(remove)
	newTask = re.sub(pattern, " ", newTask)
	arr = newTask.split(" ")
	arr = filter(None, arr)
	
	arr = [word for word in arr if word not in stopwords.words('english')]

	return arr

def extract_key_phrases_tf(fileString, date, results_tf, min=2, percentage=5):
    words = clean(fileString)
    counts = Counter(words)
    total = len(words)
    numKeywords = round(percentage / 100 * total)

    counts = Counter({x:counts[x] for x in counts if counts[x] > min-1})
    topn = counts.most_common(numKeywords)
    
    result = [x[0] for x in topn]
    results_tf.append((date,result))

def extract_key_phrases(fileString, date, results_tr):

    textRankResult = textrank.extract_key_phrases(fileString)

    #print(ranked)
    #print(values)
    result = []

    for word in textRankResult:
        strippedWord = word.replace("-", " ").replace("_"," ").strip()
        strippedWord = re.sub(r'[^a-zA-Z\s]+', '', strippedWord).strip()


        swords = strippedWord.split(" ")
        if(len(swords) < 2):
            if(len(strippedWord) <= 1):
                continue
            result.append(strippedWord)
        else:
            string = ""
            for sword in swords:
                if(len(sword) <= 1):
                    continue
                string += " " + sword
            if string.strip() != "":
                result.append(string.strip())
    
    lemmatizer = WordNetLemmatizer()
    ranked = np.asarray([lemmatizer.lemmatize(x.lower()) for x in result])
    ranked = np.asarray([x.lower() for x in result])

    keywordString = " ".join(ranked)
    results_tr.append((date, keywordString))

'''
    wordArray = fileString.split()


    result = []

    for word in wordArray:
            strippedWord = re.sub(r'[^a-zA-Z]+', '', word).strip()

            if(len(strippedWord) <= 1):
                continue
            elif(strippedWord.lower() in dictionary):
                result.append(strippedWord)


    lemmatizer = WordNetLemmatizer()
    resultString = " ".join(result)

    #extract the keywords
    ranked = textrank.extract_key_phrases(resultString)
    #find root words of keywords
    ranked = np.asarray([lemmatizer.lemmatize(x.lower()) for x in ranked])

    #eliminate duplicate keywords
    _, idx = np.unique(ranked, return_index=True)
    ranked = ranked[np.sort(idx)]

    with open(file, "w") as myfile:
        for keyword in ranked:
            myfile.write("%s\n" % keyword)
'''
def ocr(filename, date, results, results_tr):

    session = subprocess.Popen(['./ocr2.sh',filename], stdout=PIPE, stderr=PIPE)
    session.wait()
    textFile = filename.replace(".png", ".txt")

    with open(textFile, 'r') as myfile:
        result = myfile.read().replace("\\n","\n")
        results.append((date, result))
        #extract_key_phrases(result, date, results_tr)

    os.remove(textFile)
    print(date)

def getFilesToProcess(files):

    results = []

    for file in files:
        if file.endswith(".png") == False:
            continue
        date = file.split(".")[0]
        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

        results.append((date, file))

    return results

def run(path, sample_freq=1, numThreads=4):
    print(path)
    filesWithDates = getFilesToProcess(os.listdir(path))
    filesWithDates.sort(key=lambda x: x[0])
    files = [path + "/" + x[1] for x in filesWithDates]
    dates = [x[0] for x in filesWithDates]

    i = 0

    with Manager() as manager:

        results = manager.list()
        results_tr = manager.list()

        pool = Pool(processes=numThreads)

        while(i < len(files)):
            pool.apply_async(ocr, (files[i], dates[i], results, results_tr))
            i += sample_freq
        
        pool.close()
        pool.join()

        with open('fulltext.pkl', 'wb') as f:
            results = [x for x in results]
            results.sort(key=lambda x: x[0])
            pickle.dump(results, f)
'''
        with open('textranked.pkl', 'wb') as f:
            results_tr = [x for x in results_tr]
            results_tr.sort(key=lambda x: x[0])
            pickle.dump(results_tr, f)
'''


if __name__ == '__main__':
    if(len(sys.argv) <= 1):
        run(os.getcwd())
    elif(len(sys.argv) == 2):
        run(os.path.join(os.getcwd(), sys.argv[1]))
    else:
        raise("Too many arguments")
