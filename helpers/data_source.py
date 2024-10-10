from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('mode.chained_assignment', None)

def get_data(dataset, train_size):
    '''
    Get data from UCI repository
    Parameters:
        dataset: str: name of the dataset
        train_size: float: size of the training set
    Returns:
        x_train: pd.DataFrame: dataset features
        y_train: pd.DataFrame: dataset targets
    '''
    true_value = None
    match dataset:
        case 'breast': 
            repo = fetch_ucirepo(id=17)
            true_value = 'B'
        case 'blood':
            repo = fetch_ucirepo(id=176)
        case 'haberman':
            repo = fetch_ucirepo(id=43)
        case 'liver':
            repo = fetch_ucirepo(id=225)
            true_value = 1
        case 'ionosphere':
            repo = fetch_ucirepo(id=52)
        case 'parkinsons':
            repo = fetch_ucirepo(id=174)
        case 'vertebral':
            repo = fetch_ucirepo(id=212)
        case _:
            raise ValueError('Misspelling much, aer we?')

    y_name = repo.data.targets.columns[0]
    if true_value:
        repo.data.targets[y_name] = pd.Series(np.where(repo.data.targets[y_name] == true_value, 1, 0))   #Convert to binary classes

    data = clear_data(repo)
    data_train, data_test = train_test_split(data, train_size = train_size)

    return data_train.drop(y_name, axis = 1), data_test.drop(y_name, axis = 1), data_train[y_name], data_test[y_name]


def clear_data(repo: object) -> pd.DataFrame:
    if np.any(repo.data.features.dtypes == object) or np.any(repo.data.features.dtypes == str):
        repo.data.features = repo.data.features.select_dtypes(['number'])
        print("WARNING\nAt least one column has heen dropped due to being non-numeric value.")

    data = pd.concat([repo.data.features, repo.data.targets], axis=1)
    if data.isnull().values.any():
        data.dropna(inplace = True)
        print("WARNING\nSome records have been removed due to them containing NA values.")
    return data