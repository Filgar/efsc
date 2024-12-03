import numpy as np
import pandas as pd
import sys

sys.path.append('./../optimizers')
import benchmark as Benchmark
from optimizer import Optimizer

'''
Wrapper for evolving either new dataset, or single feature
'''
def evolve_new_feature(optimizer: Optimizer, x_test, name="evolved_feature", verbose=False,
                        repeats=10, epochs=50, heuristics=Benchmark.dtree_accuracy):
    '''
    Create new feature for given optimizer and dataset
    Parameters:
        optimizer: Optimizer: optimizer object
        x_test: pd.DataFrame: test dataset
        name: str: name of new feature
        verbose: bool: print logs
        repeats: int: number of repeats for heuristics function, from which mean metric is calculated
        epochs: int: number of generations
        heuristics: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int], float]: heuristics function
    Returns:
        pd.DataFrame: train dataset with new feature
        pd.DataFrame: validation dataset with new feature
        pd.DataFrame: test dataset with new feature
    '''
    x_train, _, x_val, _ = optimizer.get_training_data()
    best_feature_func, _ = optimizer.evolve_new_feature(epochs=epochs, heuristics=heuristics, verbose=verbose, repeats=repeats)

    x_test_new = x_test.assign(**{name: x_test.apply(lambda row: best_feature_func(*row), axis=1)})
    x_train_new = x_train.assign(**{name: x_train.apply(lambda row: best_feature_func(*row), axis=1)})
    x_val_new = x_val.assign(**{name: x_val.apply(lambda row: best_feature_func(*row), axis=1)})

    return x_train_new, x_val_new, x_test_new


def evolve_new_feature_set(optimizer_constructor, x_train, x_val, y_train, y_val, x_test, verbose=False,
                        repeats=10, epochs=50, heuristics=Benchmark.dtree_accuracy, population_size=64,
                        min_features=1, max_features=42):
    '''
    Create new dataset containing only evolved features
    Parameters:
        optimizer_constructor: Callable[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] -> Optimizer: optimizer constructor
        x_train: pd.DataFrame: train dataset
        x_val: pd.DataFrame: validation dataset
        verbose: bool: print logs
        repeats: int: number of repeats for heuristics function, from which mean metric is calculated
        epochs: int: number of generations
        heuristics: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int], float]: heuristics function
    Returns:
        pd.DataFrame: train dataset with new feature
        pd.DataFrame: validation dataset with new feature
        pd.DataFrame: test dataset with new feature
    '''
    x_train_new = pd.DataFrame({})
    x_test_new = pd.DataFrame({})
    x_val_new = pd.DataFrame({})

    old_metric = float('-inf')
    new_metric = float('inf')
    feature_count = 0
    epochs_without_gain = 0

    while epochs_without_gain < 3 and feature_count < min_features and feature_count < max_features:
        old_metric = new_metric
        feature_count += 1
        optimizer = optimizer_constructor(pd.concat([x_train, x_train_new], axis=1), pd.concat([x_val, x_val_new], axis=1), y_train, y_val)
        best_feature_func, _ = optimizer.evolve_new_feature(epochs=epochs, heuristics=heuristics, verbose=verbose, repeats = repeats,
                                                            target_train = x_train_new, target_test = x_val_new, population_size = population_size)
        
        x_train_new[f'evolved_feature_{feature_count}'] = pd.concat([x_train, x_train_new], axis=1).apply(lambda row: best_feature_func(*row), axis=1)
        x_test_new[f'evolved_feature_{feature_count}'] = pd.concat([x_test, x_test_new], axis=1).apply(lambda row: best_feature_func(*row), axis=1)
        x_val_new[f'evolved_feature_{feature_count}'] = pd.concat([x_val, x_val_new], axis=1).apply(lambda row: best_feature_func(*row), axis=1)
        
        new_metric = np.round(heuristics(x_train_new, x_val_new, y_train, y_val, repeats) * 100, 2)
        
        if new_metric - old_metric > 0:
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1

    return x_train_new, x_val_new, x_test_new