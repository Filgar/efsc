from abc import abstractmethod
from typing import Callable
import pandas as pd

class Optimizer:
    def __init__(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    @abstractmethod
    def optimize(self,
                pop_size: int,
                epochs: int,
                heuristics: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], float],
                **kwargs):
        raise NotImplementedError