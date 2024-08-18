import os
import pickle
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "CapstoneHearAllModelsExperiment"
mlflow.set_tracking_uri("http://mlflow:5000")
# mlflow.set_experiment(EXPERIMENT_NAME)
# mlflow.sklearn.autolog(disable=True)


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


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
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    run_id_best_model = best_run.info.run_id
    name_best_model = best_run.data.tags.get("model_name", "Unnamed_Model")
    print(f'{name_best_model}: {run_id_best_model}')
    model_uri = f"runs:/{run_id_best_model}/models"
    mlflow.register_model(model_uri, name="Heart-best-model-new")

    try:
        #Load Artifact from Mlflow
        artifact_path = "scaler/scaler.bin"
        artifact_local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id_best_model}/{artifact_path}"
        )
        with open(artifact_local_path, "rb") as f:
            scaler = pickle.load(f)
        os.makedirs("scaler", exist_ok=True)
        with open(f'scaler/best_model_scaler_{name_best_model}.bin', 'wb') as f_out:
            pickle.dump(scaler, f_out)
    except:
        print("scaler/scaler.bin downloading error")
        

    # path = client.download_artifacts(run_id=run_id_best_model, path='scaler/scaler.bin')

    # print(path)

    # # Save Artifact for best model to local file system
    # os.makedirs("scaler", exist_ok=True)
    # with open(f'scaler/best_model_scaler_{name_best_model}.bin', 'wb') as f_out:
    #     pickle.dump(path, f_out)

    logged_model_url = f'runs:/{run_id_best_model}/models'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model_url)

    # Save best model to local file system
    os.makedirs("models", exist_ok=True)
    with open(f'models/best_model_{name_best_model}.bin', 'wb') as f_out:
        pickle.dump(loaded_model, f_out)




