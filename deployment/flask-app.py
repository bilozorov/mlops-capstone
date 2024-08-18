import os
import pandas as pd
import pickle
import requests
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify


RUN_ID = os.getenv('RUN_ID', 'f147d40caf35451fa33912a2fd272c6d')
MLFLOW_SERVER_URL = os.getenv('RUN_ID', 'http://mlflow:5000')
# MLFLOW_SERVER_URL = os.getenv('RUN_ID', 'http://0.0.0.0:5001')  # for local macos system


def is_mlflow_server_running(mlflow_server_url):
    try:
        response = requests.get(mlflow_server_url)
        # Check if the server responds with a status code indicating success
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print(f"MLflow server check failed: {e}")
        return False


def get_model_scaler(mlflow_server_url, run_id):
    if is_mlflow_server_running(mlflow_server_url):
        print("MLflow server is up and running!")
        mlflow.set_tracking_uri(mlflow_server_url)

        #Load Artifact from Mlflow
        artifact_path = "scaler/scaler.bin"
        artifact_local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{artifact_path}"
        )

        with open(artifact_local_path, "rb") as f:
            loaded_scaler = pickle.load(f)

        #Load Moidel from Mlflow
        logged_model_url = f'runs:/{run_id}/models'
        loaded_model = mlflow.pyfunc.load_model(logged_model_url)
    else:
        print("MLflow server is not accessible.")
        print("Using a previously saved model and scaler.")

        model_path = "best_model_KNN.bin"
        scaler_path = "best_model_scaler_KNN.bin"

        # Load the model from the file
        with open(model_path, "rb") as file:
            loaded_model = pickle.load(file)

        model_path = "best_model_scaler_KNN"

        # Load the scaler from the file
        with open(scaler_path, "rb") as file:
            loaded_scaler = pickle.load(file)
    
    return loaded_model, loaded_scaler


def prepare_features(data, scaler):
    scaled_data = scaler.transform(data)
    return scaled_data


def get_prediction(features, model):
    prediction = model.predict(features)
    return float(prediction[0])


app = Flask('heart-attack-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    loaded_model, loaded_scaler = get_model_scaler(MLFLOW_SERVER_URL, RUN_ID)

    data_json = request.get_json()
    data = pd.DataFrame([data_json])
    

    features = prepare_features(data, loaded_scaler)
    prediction = get_prediction(features, loaded_model)

    result = {
        'prediction': prediction,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    