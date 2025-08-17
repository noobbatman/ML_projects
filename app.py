from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html exists in templates folder

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Ensure home.html exists in templates folder
    else:
        try:
            # Ensure all fields are being sent from the form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                # Adding error handling for float conversion
                reading_score=float(request.form.get('reading_score', 0)),  # Default to 0 if no value
                writing_score=float(request.form.get('writing_score', 0))   # Default to 0 if no value
            )
            
            # Get data as DataFrame
            pred_df = data.get_data_as_data_frame()
            print(pred_df)  # Debugging, can be removed later

            # Prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(features=pred_df)

            # Return results to home.html template
            return render_template('home.html', results=results[0])  # Ensure home.html displays the result

        except Exception as e:
            # Handle error if data is not correct or any issues in prediction
            print(f"Error occurred: {e}")
            return render_template('home.html', results="Error in prediction. Please check your input.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
