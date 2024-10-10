import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def knn_accuracy(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> float:
    knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
    print(y_test.dtypes)
    knn.fit(x_train, y_train)
    return knn.score(x_test, y_test)


