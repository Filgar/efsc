import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



def knn_accuracy(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> float:
    knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
    knn.fit(x_train, y_train)
    return knn.score(x_test, y_test)


def dtree_accuracy(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> float:
    scores = []
    for _ in range(10):
        dtree = DecisionTreeClassifier()
        dtree.fit(x_train, y_train)
        scores.append(dtree.score(x_test, y_test))
    return np.mean(scores)



