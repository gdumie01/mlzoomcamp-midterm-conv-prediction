import pickle
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify

def predict_single(visitor, dv, model):
    X = dv.transform([visitor])
    features = dv.get_feature_names_out()
    dX = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(dX)
    return y_pred[0]


with open('model.bin', 'rb') as f_in:
  dv, model = pickle.load(f_in)

app = Flask('predict') # give an identity to your web service

@app.route('/predict', methods=['POST']) # use decorator to add Flask's functionality to our function
def predict():
    visitor = request.get_json()
    prediction = predict_single(visitor, dv, model)
    conversion = prediction >= 0.5
    
    result = {
        'conversion_probability': float(prediction),
        'conversion': bool(conversion),
    }

    return jsonify(result)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696