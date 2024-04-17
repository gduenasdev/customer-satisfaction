from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/Users/gduenas/Desktop/duendev/development/customer-satisfaction/data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:/Users/gduenas/Library/Application Support/zenml/local_stores/40fc01cd-90af-416a-95bc-fba2027f5f85/mlruns"