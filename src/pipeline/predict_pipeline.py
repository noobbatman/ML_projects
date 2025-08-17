import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Before Loading Model and Preprocessor")
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Log model and preprocessor paths
            logging.info(f"Model Path: {model_path}")
            logging.info(f"Preprocessor Path: {preprocessor_path}")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("After Loading Model and Preprocessor")

            # Process data
            data_scaled = preprocessor.transform(features)
            logging.info(f"Data after transformation: {data_scaled}")

            # Make prediction
            preds = model.predict(data_scaled)
            logging.info(f"Prediction result: {preds}")

            return preds

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise CustomException(e, sys)



class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, 
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Creating a dictionary for the data to be converted into a DataFrame
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert dictionary to pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
