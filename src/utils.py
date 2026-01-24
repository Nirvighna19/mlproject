import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path , obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , "wb") as file_obj:
            dill.dump(obj , file_obj)

    except Exception as e:
        raise CustomException(e , sys)
    
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def evaluate_model(xtrain, ytrain, xtest, ytest, models, param):
    report = {}

    for model_name, model in models.items():
        params_grid = param.get(model_name, {})

        # 1️⃣ Decide which estimator to use
        if model_name == "CatBoosting Regressor":
            regressor = model

        elif params_grid and len(params_grid) > 0:
            gs = GridSearchCV(
                estimator=model,
                param_grid=params_grid,
                cv=3,
                n_jobs=-1
            )
            gs.fit(xtrain, ytrain)
            regressor = gs.best_estimator_

        else:
            regressor = model

        # 2️⃣ ALWAYS fit (no exceptions)
        regressor.fit(xtrain, ytrain)

        # 3️⃣ Predict only AFTER fitting
        ytrain_pred = regressor.predict(xtrain)
        ytest_pred = regressor.predict(xtest)

        test_score = r2_score(ytest, ytest_pred)
        report[model_name] = test_score

    return report
