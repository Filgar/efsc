import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score



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

def regressor_r2_score(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> float:
    # Initialize the regressor (Decision Tree Regressor)
    scores = []
    for _ in range(10):
        regressor = DecisionTreeRegressor()
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        scores.append(r2_score(y_test, y_pred))
    return np.mean(scores)



