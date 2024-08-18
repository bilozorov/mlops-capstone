import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


import sklearn
# Print the version of scikit-learn
print("Scikit-learn version:", sklearn.__version__)


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


import mlflow
EXPERIMENT_NAME = "CapstoneHearAllModelsExperiment"
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(disable=True)


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    
    X_transformed, y, scaler = data

    os.makedirs("scaler", exist_ok=True)
    with open('scaler/scaler.bin', 'wb') as f_out : 
        pickle.dump(scaler, f_out)


    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    models = {
        'Logistic_Regression': LogisticRegression(),
        'Decision_Tree': DecisionTreeClassifier(),
        'Random_Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Gradient_Boosting': GradientBoostingClassifier(),
        # 'XGBoost': XGBClassifier(),
        'AdaBoost': AdaBoostClassifier(), 
        'Naive_Bayes': GaussianNB(),       
        'MLP_Neural_Network': MLPClassifier() 
    }


    for name, model in models.items():
        # print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        os.makedirs("models", exist_ok=True)
        with open(f"models/model_{name}.bin", "wb") as f_out:
            pickle.dump(model, f_out)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)     

        # Log with MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                model,
                artifact_path="models",
                registered_model_name=name  # Specify the model name for the registry
            )
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_artifact("scaler/scaler.bin", artifact_path="scaler")
            mlflow.set_tag("model_name", name)  # Optionally, add a tag with the model name









