import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from mlflow import MlflowClient

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

print("Loading models from MLFlow registry")

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://host.docker.internal:5000")

def load_latest_model(model_name):
    """Load the latest version of a model from the Model Registry."""
    try:
        client = MlflowClient()
        latest_version = client.search_model_versions(f"name='{model_name}'")[0].version
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None
    
def print_model_details(model_name, model):
    """Print details about the loaded model."""
    print(f"\nModel: {model_name}")
    print(f"Type: {type(model)}")
    print(f"Flavor: {model.metadata.flavors}")
    print(f"Run ID: {model.metadata.run_id}")

def make_predictions(model, X):
    """Make predictions using the loaded model."""
    return model.predict(X)

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# List of models to load
models_to_load = [
    "Logistic_Regression_Model",
    "Random_Forest_Model",
    "SVM_Model"
]


# Load models, print details, and make predictions
for model_name in models_to_load:
    try:
        model = load_latest_model(model_name)
        print_model_details(model_name, model)
        
        # Make predictions
        predictions = make_predictions(model, X)
        print(f"Predictions shape: {predictions.shape}")
        print(f"First 5 predictions: {predictions[:5]}")
        
    except Exception as e:
        print(f"Error loading or using model {model_name}: {str(e)}")

print("\nAll models loaded and predictions made.")



