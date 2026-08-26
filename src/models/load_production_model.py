import mlflow


MODEL_URI = (
    "models:/rossmann-sales-random-forest/Production"
)


def main():

    print("Loading Production model from MLflow...")

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print("✓ Production model loaded successfully")

    print(f"Model URI: {MODEL_URI}")
    print(f"Model type: {type(model)}")

    print("\n================================")
    print("PRODUCTION MODEL LOAD COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()