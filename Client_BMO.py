import warnings
import flwr as fl
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss

import utils

# Load dataset
i = 1
(X_train_1,y_train_1) = utils.load_credit_train_i_Default(i)
(X_test, y_test,y_test_p_1) = utils.load_credit_test_i_Default(i)

# Create LogisticRegression Model
model = LogisticRegressionCV(cv=5, random_state=0,
                             penalty="l2",
                             max_iter=utils.load_parameter_C()[0],  # local epoch
                             Cs=utils.load_parameter_C()[1]
                             )

# Setting initial parameters, akin to model.compile for keras models
utils.set_initial_params(model)

class MnistClient(fl.client.NumPyClient):
    def get_parameters(self): # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config): # type: ignore
        utils.set_model_params(model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_1, y_train_1)
            print(f"Training finished for round {config['rnd']}")

        accuracy = model.score(X_test, y_test)
        print(accuracy)
        profit = utils.profit_evaluation(model.predict(X_test), y_test_p_1)
        print(profit)

        return utils.get_model_parameters(model), len(X_train_1), {}

    def evaluate(self, parameters, config): # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

fl.client.start_numpy_client("0.0.0.0:8080", client=MnistClient())