import mlflow
from mlflow import MlflowClient


RUN_ID = "edbcc6a668884302a6eabd71ea0c5f72"

MODEL_NAME = "rossmann-sales-random-forest"


def main():

    print("Connecting to MLflow...")

    client = MlflowClient()

    model_uri = f"runs:/{RUN_ID}/random_forest_model"

    print("Registering model...")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("\n✓ Model registered")

    print(f"Model name: {registered_model.name}")
    print(f"Model version: {registered_model.version}")

    print("\n================================")
    print("MODEL REGISTRATION COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()