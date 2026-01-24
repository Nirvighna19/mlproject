import os
import sys
import numpy as np
import pandas as pd
import dill
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
    
def evaluate_model(xtrain , ytrain, xtest ,ytest , models):

    try:
        report={}

        for model_name , model in models.items():
                regressor=model
                regressor.fit(xtrain , ytrain)

                ytrain_pred=regressor.predict(xtrain)
                ytest_pred=regressor.predict(xtest)

                train_model_score=r2_score(ytrain , ytrain_pred)
                test_model_score=r2_score(ytest , ytest_pred)

                report[model_name]=test_model_score

        return report
    except Exception as e:
         raise CustomException(e , sys)