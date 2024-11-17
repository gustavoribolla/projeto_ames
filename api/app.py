from flask import Flask, request, jsonify
import pathlib
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

with open('C:/Insper_real_oficial/machine_learning/projeto_ames/model/model.pkl', 'rb') as f:
    model = pickle.load(f)



@app.route('/teste', methods=['GET'])
def teste():
    return jsonify({'message': 'Hello, World!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        data = pd.DataFrame(data , index=[0])
        prediction = model.predict(data)
        print("Prediction:", prediction)
        prediction_list = prediction.tolist()
        return jsonify({'prediction': prediction_list}), 200
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True)
