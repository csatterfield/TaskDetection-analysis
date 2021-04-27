import numpy as np
import pandas as pd
from scipy.optimize import minimize



def weighted_average(weights, df):
    cols = df.columns
    sc_w2v = [x for x in cols if "sc_w2v" in x]
    wt_w2v = [x for x in cols if "wt_w2v" in x]
    sc_rake = [x for x in cols if "sc_rake" in x]
    wt_tf = [x for x in cols if "wt_tf" in x]

    df[sc_w2v] = df[sc_w2v] * weights[0]
    df[wt_w2v] = df[wt_w2v] *  weights[1]
    df[sc_rake] = df[sc_rake] * weights[2]
    df[wt_tf] = df[wt_tf] * weights[3]

    return average(df)

def average(df):
    cols = df.columns
    c1 = [x for x in cols if "1" in x]
    c2 = [x for x in cols if "2" in x]
    c3 = [x for x in cols if "3" in x]
    c4 = [x for x in cols if "4" in x]
    c5 = [x for x in cols if "5" in x]
    c6 = [x for x in cols if "6" in x]

    v1 = np.mean(df[c1],axis=1)
    v2 = np.mean(df[c2],axis=1)
    v3 = np.mean(df[c3],axis=1)
    v4 = np.mean(df[c4],axis=1)
    v5 = np.mean(df[c5],axis=1)
    v6 = np.mean(df[c6],axis=1)



    a = np.array([v1,v2,v3,v4,v5,v6]).transpose()
    pred = np.argmax(a, axis=1) + 1

    df = df[["author", "expected", "participant"]]
    df["predicted"] = pred

    correct = (df["predicted"] == df["expected"]).astype(int)
    df["correct"] = correct
    df.to_excel("matching_ensemble.xlsx")
    return np.mean(correct)


if __name__ == "__main__":
    df = pd.read_excel("confidences.xlsx")
    print(weighted_average([1,1,0.8,0.1], df))
    #res = minimize(weighted_average, [0.7,0,0.3,0], args=(df), method="Powell")
    #print(res.status, res.message)
    #print(res.x)

    score = 0
    best = None
    count = 0
