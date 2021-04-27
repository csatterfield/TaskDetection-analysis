from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import warnings
import tensorflow as tf
from sklearn.model_selection import KFold
#warnings.simplefilter("ignore")



def softargmax(x, beta=12):
  x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*np.exp(beta)) * x_range, axis=1)

def custom_loss(i):
    def loss(y_true, y_pred):
        w = tf.reshape(i, [tf.shape(i)[0], 4, 6])
        w_y = tf.multiply(w, tf.reshape(y_pred, [tf.shape(i)[0], 4, 1]))
        avg = tf.reduce_mean(w_y,1)
        pred_labels = softargmax(avg) + 1
        pred_labels = tf.cast(pred_labels, "float64")
        return tf.keras.losses.categorical_crossentropy(y_true, pred_labels)
    return loss

def predict(X, weights):
    X = X.reshape(X.shape[0], 4, 6)
    weights = weights.reshape(weights.shape[0], 4, 1)
    avg = np.mean(np.multiply(X, weights), axis=1)
    print(np.argmax(avg, axis=1) + 1)



def main():
    i = tf.keras.layers.Input((24,), dtype="float64")
    x = tf.keras.layers.Dense(256, activation='relu', dtype="float64")(i)
    o = tf.keras.layers.Dense(4, activation='sigmoid', dtype="float64")(x)
    model = tf.keras.models.Model(i,o)

    cl = custom_loss(i)

    model.compile(loss=cl, experimental_run_tf_function=False)

    df = pd.read_excel("confidences.xlsx")

    partipicants = list(df["participant"].unique())
    X = df.values[:,3:]
    y = df["expected"].values

    kfold = KFold(n_splits=10)

    for train_index, test_index in kfold.split(X):
        model.fit(X[train_index], y[train_index], epochs=10, batch_size=32)
        y_pred = predict(X[test_index], model.predict(X[test_index]))



if __name__ == "__main__":
    main()