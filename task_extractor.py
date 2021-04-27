import pickle
import pandas as pd
from datetime import datetime, date, time
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textrank
import os
import sys
import numpy as np
stopwords = stopwords.words("english")

def valid(dateobj, start, end):
    return dateobj >= start and dateobj < end        

def widf(task_counters):
    idf_counters = []
    for counter in task_counters:
        n = Counter()
        for key in counter:
            n[key] = 1
        idf_counters.append(n)
    total_counter = sum(idf_counters, Counter())

    weighted_counters = []
    for counter in task_counters:
        weighted = {}
        for key in counter.keys():
            weighted[key] = counter[key]/total_counter[key]
        weighted_counters.append(weighted)
    return weighted_counters


class ScreenshotTaskExtractor(object):

    def __init__(self, vocab):
        self.vocab = vocab


    def isProbablyEmail(self, task):
        excludedWords = ['compose', 'gmail', 'inbox', 'google', 'starred', 'sent','mail','drafts','more','terms','privacy','program','policies']
        i = 0
        for word in excludedWords:
            if word in task:
                i += 1
        return i > 4

    def get_tasks_for_participant(self, path_to_data, participant, without_emails=True, ungrouped=False, filter=lambda x: word_tokenize(x), using_widf=False, drop_empty_rows=True):

        with open(f"{path_to_data}/{participant}/fulltext.pkl", "rb") as f:
            snapshotsWithDates = pickle.load(f)
        
        df = pd.read_excel(f"{path_to_data}/{participant}/taskswitches_annotated.xlsx")
        offset = df[df["task"] == "offset"]["end"].iloc[0]
        df = df[df["task"] != "offset"]
        task_words_ungrouped = []
        task_order = []

        studyStartTime = snapshotsWithDates[0][0] - (datetime.combine(date.min, offset) - datetime.min)

        for _,row in df.iterrows():
            startDelta = datetime.combine(date.min, row["start"]) - datetime.min 
            endDelta = datetime.combine(date.min, row["end"]) - datetime.min
            start = studyStartTime + startDelta
            end = studyStartTime + endDelta

            snapshotsInTask = [x[1].lower() for x in snapshotsWithDates if valid(x[0], start, end)]

            if(without_emails):
                snapshotsInTask = [x for x in snapshotsInTask if not self.isProbablyEmail(x)]

            words = []
            for snapshot in snapshotsInTask:
                tokens = filter(snapshot)
                snapshot_words = [x for x in tokens if len(x) > 2]
                snapshot_words = [x for x in snapshot_words if x in self.vocab]
                words.extend(snapshot_words)

            task_words_ungrouped.append(words)
            task_order.append(row["task"])

        if(drop_empty_rows):
            task_words_ungrouped = [x for x in task_words_ungrouped if len(x) > 0]
        
        task_counters = [Counter(x) for x in task_words_ungrouped]
        if(not ungrouped):
            counters = defaultdict(Counter)
            for counter, task in zip(task_counters, task_order):
                counters[task] += counter
            order, task_counters = zip(*counters.items())

        if(using_widf):
            task_counters = widf(task_counters)

        for task in task_counters:
            norm = np.linalg.norm(list(task.values()))
            for word in task:
                task[word] = task[word]/norm

        return task_counters, task_order

class WindowTitleTaskExtractor:

    def get_tasks_for_participant(self, path_to_data, participant, without_emails=True, ungrouped=False, filter=lambda x: word_tokenize(x), widf=False):
        
        
        
        raise("Not yet implemented")
        
        
        
        
        df = pd.read_excel(f"{path_to_data}/{participant}/taskswitches_annotated.xlsx")
        appdata = pd.read_csv(f"{path_to_data}/{participant}/appdata_fixed.csv")

        offset = df[df["task"] == "offset"]["end"].iloc[0]
        df = df[df["task"] != "offset"]
        tasks = defaultdict(list)
        tasks_ungrouped = []

        studyStartTime = appdata["StartTime"][0] - (datetime.combine(date.min, offset) - datetime.min).total_seconds()

        for _,row in df.iterrows():
            startDelta = datetime.combine(date.min, row["start"]) - datetime.min
            endDelta = datetime.combine(date.min, row["end"]) - datetime.min
            start = studyStartTime + startDelta.total_seconds()
            end = studyStartTime + endDelta.total_seconds()

            titles = appdata[((appdata["StartTime"] >= start) & (appdata["StartTime"] < end)) | ((appdata["EndTime"] >= start) & (appdata["EndTime"] < end))]
            titles = list(titles["WindowTitle"])
            titles = [x.lower() for x in titles]

            if(without_emails):
                snapshotsInTask = [x for x in titles if not "gmail" in x]

            tasks[row["task"]].extend(snapshotsInTask)
            tasks_ungrouped.append({"snapshots": snapshotsInTask, "task":row["task"], "duration": end-start }   )

        if(ungrouped):
            return filter(tasks_ungrouped)
        return filter([{"task": x, "snapshots": tasks[x], "duration":0 } for x in tasks.keys()])
