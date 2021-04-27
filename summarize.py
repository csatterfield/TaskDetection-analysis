import pickle
import pandas as pd
from datetime import datetime, date, time
from collections import defaultdict
import textrank
import os
import sys

def valid(dateobj, start, end):
    return dateobj >= start and dateobj < end

def extract(taskString):
    keyphrases = textrank.extract_key_phrases(taskString)
    return keyphrases

def extract_task_without_email(taskList):
    taskListFiltered = [extract(x) for x in taskList if not isProbablyEmail(x)]
    return taskListFiltered

def isProbablyEmail(task):
    excludedWords = ['Compose', 'Gmail', 'Inbox', 'Google', 'Starred', 'Sent','Mail','Drafts','More','Terms','Privacy','Program','Policies']
    i = 0
    for word in excludedWords:
        if word in task:
            i += 1
    return i > 4

def run():
    with open("fulltext.pkl", "rb") as f:
        snapshotsWithDates = pickle.load(f)
    
    df = pd.read_excel("taskswitches_annotated.xlsx")
    tasks = defaultdict(list)
    tasks_ranked = defaultdict(list)

    studyStartTime = snapshotsWithDates[0][0]

    for _,row in df.iterrows():
        startDelta = datetime.combine(date.min, row["start"]) - datetime.min
        endDelta = datetime.combine(date.min, row["end"]) - datetime.min
        start = studyStartTime + startDelta
        end = studyStartTime + endDelta

        snapshotsInTask = [x[1] for x in snapshotsWithDates if valid(x[0], start, end)]
        tasks[row["task"]].extend(snapshotsInTask)
    
    for task in tasks:
        #tasks_ranked[task] = [extract(x) for x in tasks[task]]
        tasks_ranked[task] = extract_task_without_email(tasks[task])
    
    with open("ranked_filtered.pkl", "wb") as f:
        pickle.dump(tasks_ranked, f)


if __name__ == '__main__':

    if(len(sys.argv) == 2):
        os.chdir(sys.argv[1])

    run()