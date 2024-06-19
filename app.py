import pickle
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.fertilizer import fertilizer_dic

# Load the pre-trained model
model_path = 'Models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(model_path, 'rb'))

# Define the Flask app and Flask-RESTful Api
app = Flask(__name__)
api = Api(app)

class CropPrediction(Resource):
    def post(self):
        # Extract data from the request
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create input array for the model
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Make prediction
        prediction = crop_recommendation_model.predict(data)
        final_prediction = prediction[0]

        # Return the prediction as a JSON response
        return {'prediction': final_prediction}, 200

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv(r'C:\Users\salah\Downloads\NPK reco\app\Data\fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]

    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = fertilizer_dic[key]
    return jsonify({'recommendation': Markup(str(response))})

# Add the CropPrediction resource to the API
api.add_resource(CropPrediction, '/crop-predict')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
