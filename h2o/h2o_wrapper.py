import pandas as pd
import numpy as np

import h2o
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator, H2OGeneralizedLinearEstimator
from h2o.automl import H2OAutoML


import warnings
warnings.filterwarnings("ignore", module='h2o')

# REQUIRES h2o SERVER RUNNING
h2o.no_progress()

def gradient_boost(features, target, train, val, test, y_test):
    gbm = H2OGradientBoostingEstimator(
        ntrees=100,
        max_depth=5,
        min_rows=10,
        learn_rate=0.1,
        seed=623,
    )

    gbm.train(x=features, y=target, training_frame=train, validation_frame=val, verbose=False)

    predictions = np.round(gbm.predict(test).as_data_frame().values.flatten())
    return np.sum(predictions == y_test.values) / len(y_test.values)



def random_forest(features, target, train, val, test, y_test):
    rf = H2ORandomForestEstimator(
        ntrees=100,
        max_depth=5,
        min_rows=10,
        seed=623,
    )

    rf.train(x=features, y=target, training_frame=train, validation_frame=val, verbose=False)

    predictions = np.round(rf.predict(test).as_data_frame().values.flatten())
    return np.sum(predictions == y_test.values) / len(y_test.values)



def generalized_linear(features, target, train, val, test, y_test, solver='auto'):
    glm = H2OGeneralizedLinearEstimator(
        alpha=0.5,
        lambda_=0.001,
        seed=623,
        solver=solver
    )
    glm.train(x=features, y=target, training_frame=train, validation_frame=val, verbose=False)

    predictions = np.round(glm.predict(test).as_data_frame().values.flatten())
    return np.sum(predictions == y_test.values) / len(y_test.values)



def auto_ml(features, target, train, val, test, y_test):
    aml = H2OAutoML(
        max_models=25,
        max_runtime_secs_per_model=30,
        seed=623,
        balance_classes=True,
        class_sampling_factors=[0.5, 1.25],
        verbosity=None
    )
    aml.train(x=features, y=target, training_frame=train, validation_frame=val)

    predictions = np.round(aml.leader.predict(test).as_data_frame().values.flatten())
    return np.sum(predictions == y_test.values) / len(y_test.values)