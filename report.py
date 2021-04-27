import pickle
import pandas as pd
import os
from collections import Counter, defaultdict
os.chdir("../archives")

results = defaultdict(dict)
results_nointersection = defaultdict(dict)

participant_list = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]

for p in participant_list:

    print()
    print(p)
    print()

    tasks = {}

    with open(p + "/ranked_filtered.pkl", "rb") as f:
        data = pickle.load(f)

    for i in range(1,7):
        keywords = []
        [keywords.extend(x) for x in data[i]]
        top10 = Counter(keywords).most_common(10)
        results[p][i] = ", ".join([x[0] for x in top10])
        tasks[i] = set(keywords)

    intersection = tasks[1].intersection(tasks[2], tasks[3], tasks[4], tasks[5], tasks[6])

    print()
    print("intersection")
    print(intersection)

    for i in range(1,7):
        keywords = []
        [keywords.extend(x.difference(intersection)) for x in data[i]]
        top10 = Counter(keywords).most_common(10)
        results_nointersection[p][i] = ", ".join([x[0] for x in top10])

df = pd.DataFrame(results)
df.transpose().to_excel("simpleSummarizations.xlsx", sheet_name="Simple")

df = pd.DataFrame(results_nointersection)
df.transpose().to_excel("summarizations.xlsx", sheet_name="Without Intersection")
