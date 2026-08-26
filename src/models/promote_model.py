import mlflow
from mlflow import MlflowClient


MODEL_NAME = "rossmann-sales-random-forest"
MODEL_VERSION = "2"


def main():

    print("Connecting to MLflow...")

    client = MlflowClient()

    print(
        f"Promoting {MODEL_NAME} "
        f"version {MODEL_VERSION}..."
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        stage="Production",
        archive_existing_versions=True
    )

    print("\n✓ Model promoted to Production")

    print(f"Model: {MODEL_NAME}")
    print(f"Version: {MODEL_VERSION}")
    print("Stage: Production")

    print("\n================================")
    print("MODEL PROMOTION COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()