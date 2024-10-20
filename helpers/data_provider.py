from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('mode.chained_assignment', None)

def get_train_test_data(dataset: str, train_size: float):
    '''
    Get dataset split into training and testing sets
    Parameters:
        dataset: str: name of the dataset
        train_size: float: size of the training set
    Returns:
        x_train: pd.DataFrame: train dataset features
        x_test: pd.DataFrame: test dataset features
        y_train: pd.DataFrame: train dataset targets
        y_train: pd.DataFrame: test dataset targets
    '''
    target_name, data = __get_data(dataset)
    data_train, data_test = train_test_split(data, train_size = train_size)
    return data_train.drop(target_name, axis = 1), data_test.drop(target_name, axis = 1), data_train[target_name], data_test[target_name]


def get_train_test_validation_data(dataset: str, train_size: float, validation_size: float):
    '''
    Get dataset split into training and testing sets
    Parameters:
        dataset: str: name of the dataset
        train_size: float: size of the training set
        validation_size: float: size of the validation set
    Returns:
        x_train: pd.DataFrame: train dataset features
        x_test: pd.DataFrame: test dataset features
        x_val: pd.DataFrame: validation dataset features
        y_train: pd.DataFrame: train dataset targets
        y_train: pd.DataFrame: test dataset targets
        y_val: pd.DataFrame: validation dataset targets
    '''
    target_name, data = __get_data(dataset)
    data_train, data_test = train_test_split(data, train_size = train_size)
    data_train, data_val = train_test_split(data_train, train_size = validation_size / train_size)
    
    return (data_train.drop(target_name, axis = 1), data_test.drop(target_name, axis = 1), data_val.drop(target_name, axis = 1),
            data_train[target_name], data_test[target_name], data_val[target_name])


def __get_data(dataset: str) -> pd.DataFrame:
    '''
    Get data from UCI repository
    Parameters:
        dataset: str: name of the dataset
    Returns:
        y_name: str: name of the target column
        data: pd.DataFrame: ready-to-use dataset
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
            true_value = 1
        case 'liver':
            repo = fetch_ucirepo(id=225)
            true_value = 1
        case 'ionosphere':
            repo = fetch_ucirepo(id=52)
            true_value = 'g'
        case 'parkinsons':
            repo = fetch_ucirepo(id=174)
        case 'rice':
            repo = fetch_ucirepo(id=545)
            true_value = "Osmancik"
        case 'banknote':
            repo = fetch_ucirepo(id=267)
        case _:
            raise ValueError('Misspelling much, aer we?')

    target_name = repo.data.targets.columns[0]
    if true_value:
        repo.data.targets[target_name] = pd.Series(np.where(repo.data.targets[target_name] == true_value, 1, 0))   #Convert to binary classes

    return target_name, __clear_data(repo)
    


def __clear_data(repo: object) -> pd.DataFrame:
    '''
    Clears dataset from non-numeric values and NA values
    Parameters:
        repo: object: UCI repository object
    Returns:
        data: pd.DataFrame: cleared dataset
    '''
    if np.any(repo.data.features.dtypes == object) or np.any(repo.data.features.dtypes == str):
        repo.data.features = repo.data.features.select_dtypes(['number'])
        print("WARNING\nAt least one column has heen dropped due to being non-numeric value.")

    data = pd.concat([repo.data.features, repo.data.targets], axis=1)
    if data.isnull().values.any():
        data.dropna(inplace = True)
        print("WARNING\nSome records have been removed due to them containing NA values.")
    return data