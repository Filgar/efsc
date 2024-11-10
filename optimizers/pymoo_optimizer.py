from optimizer import Optimizer
from deap import base, creator, gp, tools, algorithms
import operator
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from typing import Callable
import numpy as np
import pandas as pd
import math

def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1e-10

def protected_log(x):
    return math.log(x) if x > 0 else 1e-10

def protected_sqrt(x):
    return math.sqrt(abs(x))

def square_root(x):
    return x**2

def prepare_expression_grammar(feature_columns):
    pset = gp.PrimitiveSet("MAIN", len(feature_columns))

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(square_root, 1)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(math.tanh, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(protected_log, 1)

    # Rename arguments for readability
    for i, col_name in enumerate(feature_columns):
        pset.renameArguments(**{f"ARG{i}": col_name})

    return pset


class PymooOptimizer(Optimizer):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.pset = prepare_expression_grammar(x_train.columns)

    def evolve_new_feature(self, epochs, heuristics, verbose = True, target_train = None, target_test = None):
        target_train = self.x_train if target_train is None else target_train
        target_test = self.x_test if target_test is None else target_test

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        def eval_feature(individual):
            func = toolbox.compile(expr=individual)
            # Apply the new feature function to each row by unpacking arguments
            new_feature_train = self.x_train.apply(lambda row: func(*row), axis=1)
            new_feature_test = self.x_test.apply(lambda row: func(*row), axis=1)

            # Add new feature to dataset and evaluate accuracy
            x_train_augmented = pd.concat([target_train, new_feature_train.rename("new_feature")], axis=1)
            x_test_augmented = pd.concat([target_test, new_feature_test.rename("new_feature")], axis=1)
            return heuristics(x_train_augmented, x_test_augmented, self.y_train, self.y_test),

        toolbox.register("evaluate", eval_feature)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        population = toolbox.population(n=50)
        hall_of_fame = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run the evolution
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=epochs, stats=stats, halloffame=hall_of_fame, verbose=verbose)

        # Return the best individual found as the new feature expression
        best_feature_func = toolbox.compile(expr=hall_of_fame[0])
        return best_feature_func

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