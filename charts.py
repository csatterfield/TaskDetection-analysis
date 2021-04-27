import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

data = pd.read_excel("matching_results.xlsx")

def save(ax, filename):
    fig = ax.get_figure()
    fig.savefig(filename)
    plt.clf()

def overall_accuracy():
    results = []
    for method in data["method"].unique():
        df = data[data.method == method]
        y_true = df["expected"]
        y_pred = df["predicted"]
        results.append({"method": method, "accuracy": accuracy_score(y_true, y_pred)})

    df = pd.DataFrame(results)
    ax = sns.barplot(x="method", y="accuracy", data=df).set_title("Overall Categorization Accuracy By Method")
    save(ax, "overall_v2.png")
    
def accuracy_grouped_by_author():
    results = []
    for method in data["method"].unique():
        df = data[(data.method == method) & (data.author == "gail")]
        y_true = df["expected"]
        y_pred = df["predicted"]
        results.append({"method": method, "accuracy": accuracy_score(y_true, y_pred), "author": "gail"})

        df = data[(data.method == method) & (data.author == "thomas")]
        y_true = df["expected"]
        y_pred = df["predicted"]
        results.append({"method": method, "accuracy": accuracy_score(y_true, y_pred), "author": "thomas"})

    df = pd.DataFrame(results)
    ax = sns.barplot(x="method", y="accuracy", hue="author", data=df).set_title("Accuracy Grouped By Keywords Author")
    save(ax, "grouped_by_author_v2.png")


def accuracy_by_task():
    results = []
    for method in ["rake", "w2v"]:
        for task in [1,2,3,4,5,6]:
            df = data[(data.method == method) & (data.expected == task)]
            y_true = df["expected"]
            y_pred = df["predicted"]
            results.append({"method": method, "accuracy": accuracy_score(y_true, y_pred), "task": task})

    df = pd.DataFrame(results)
    ax = sns.barplot(x="task", y="accuracy", hue="method", data=df).set_title("Accuracy Per Task")
    save(ax, "grouped_by_task_v2.png")


def confusion_matrix_rake():
    df = data[data.method == "rake"]
    y_true = df["expected"]
    y_pred = df["predicted"]
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, cmap="BuGn", fmt="d", annot=True, linewidths=0.5, xticklabels=range(1,7),yticklabels=range(1,7)).set_title("Confusion Matrix - RAKE")
    save(ax, "cm_rake_v2.png")

def confusion_matrix_w2v():
    df = data[data.method == "w2v"]
    y_true = df["expected"]
    y_pred = df["predicted"]
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, cmap="BuGn", fmt="d", annot=True, linewidths=0.5, xticklabels=range(1,7),yticklabels=range(1,7)).set_title("Confusion Matrix - W2V")
    save(ax, "cm_w2v_v2.png")

overall_accuracy()
accuracy_grouped_by_author()
accuracy_by_task()
confusion_matrix_rake()
confusion_matrix_w2v()

