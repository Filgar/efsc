from optimizer import Optimizer

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from typing import Callable
import numpy as np
import pandas as pd

class PymooOptimizer(Optimizer):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)

    def optimize(self,
                pop_size: int,
                epochs: int,
                heuristics: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], float],
                **kwargs):
        '''
        Optimize feature selection using genetic algorithm
        Parameters:
            pop_size: int: population size
            epochs: int: number of generations
            heuristics: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], float]:
                function to optimize
            **kwargs: dict: additional parameters
                - population_provider: Sampling: population sampling strategy
                - verbose: bool: print logs
        Returns:
            np.array: selected features
            float: heuristics value
        '''

        this = self     #Good old JS trick except in Python and reversed :)

        # Dynamically create Problem derived class using provided heuristics
        class SelectFeaturesProblem(Problem):
            def __init__(self, n_var):
                super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=0, xu=1, vtype=bool) 

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = np.array([-heuristics(this.x_train.loc[:, solution], this.x_test.loc[:,  solution], this.y_train, this.y_test)
                                    if np.sum(solution) > 0 else np.inf for solution in x])

        algorithm = GA(
            sampling = kwargs.get("population_provider", BinaryRandomSampling()),
            pop_size=pop_size,
            mutation=BitflipMutation(),
            crossover=TwoPointCrossover(),
            eliminate_duplicates=True
        )

        res = minimize(SelectFeaturesProblem(self.x_train.shape[1]),
                    algorithm,
                    ('n_gen', epochs),
                    verbose=kwargs.get("verbose", False))

        return res.X, res.F