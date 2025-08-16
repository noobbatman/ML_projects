import os
import sys
from src.logger import logging
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            if not para:
                # If no hyperparameters are provided, fit the model normally
                model.fit(X_train, y_train)
                train_model_score = r2_score(y_train, model.predict(X_train))
                test_model_score = r2_score(y_test, model.predict(X_test))
                report[model_name] = test_model_score
            else:
                # Perform grid search for models with hyperparameters
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                
                # Update model with the best hyperparameters
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                # Evaluate model
                train_model_score = r2_score(y_train, model.predict(X_train))
                test_model_score = r2_score(y_test, model.predict(X_test))
                report[model_name] = test_model_score

        # Identify and log the best model
        best_model_name = max(report, key=report.get)
        best_model_score = report[best_model_name]

      

        return report

    except Exception as e:
        raise CustomException(e, sys)