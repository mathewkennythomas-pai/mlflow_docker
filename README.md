# MLFlow and Docker for Machine Learning Development

Deploying ML models using MLFlow and Docker

# MLFlow
MLflow is an open-source platform designed to manage the machine learning lifecycle, including experimentation, reproducibility, and deployment. Here’s how MLflow can be beneficial for various ML development use cases:

## 1. Experiment Tracking
MLflow allows you to log and track experiments easily. You can log parameters, metrics, and models, making it simple to compare different runs and understand which configurations yield the best results.

### Example Use Case:
- **Hyperparameter Tuning**: Track different hyperparameter settings and their corresponding model performance metrics to identify the optimal configuration.

## 2. Model Management
With MLflow, you can manage multiple models in a centralized repository. This includes versioning models, which is crucial for maintaining the integrity of your ML pipeline.

### Example Use Case:
- **Model Versioning**: Keep track of different versions of a model as you iterate on your development, ensuring that you can revert to previous versions if needed.

## 3. Reproducibility
MLflow promotes reproducibility by allowing you to log the environment in which your models were trained. This includes dependencies, versions, and configurations.

### Example Use Case:
- **Reproducing Results**: If a model performs well, you can reproduce the exact environment and code to validate the results or deploy the model confidently.

## 4. Deployment
MLflow simplifies the deployment of machine learning models to various platforms. You can deploy models as REST APIs or integrate them into existing applications seamlessly.

### Example Use Case:
- **Serving Models**: Deploy your trained models as REST APIs using MLflow’s built-in serving capabilities, making it easy to integrate into web applications.

## 5. Integration with Popular Libraries
MLflow integrates well with popular ML libraries such as Scikit-learn, TensorFlow, and PyTorch, making it versatile for different types of ML projects.

### Example Use Case:
- **Cross-Framework Compatibility**: Use MLflow to manage models trained with different libraries, ensuring a consistent workflow across your projects.

MLflow is a powerful tool for managing the machine learning lifecycle, enhancing collaboration, and improving productivity in ML development. By leveraging its capabilities, teams can streamline their workflows and focus on building better models.

![image](https://github.com/user-attachments/assets/285e21d4-f8a6-455a-a42c-ceaeceac30cb)

![image](https://github.com/user-attachments/assets/f2f9c1f5-2bfa-4dea-bc43-76aa28faa291)



# MLflow Model Registry: Logging and Inference Guide

MLflow is an open-source platform that streamlines the machine learning lifecycle, including model training, logging, and deployment. This guide will walk you through using the MLflow Model Registry to log trained models and make predictions with them.

## Prerequisites
- Python 3.x
- MLflow installed (`pip install mlflow`)
- Required libraries: `pandas`, `numpy`, `scikit-learn`

## Overview
The MLflow Model Registry allows you to:
1. Log trained models.
2. Register models with versioning.
3. Transition models between different stages (e.g., Staging, Production).
4. Load and make predictions with registered models.

## Step 1: Logging Trained Models

### Example: Logistic Regression, Random Forest, and SVM

1. **Import Libraries**:
   ```python
   import mlflow
   import mlflow.sklearn
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.svm import SVC
   from sklearn.metrics import accuracy_score
   ```

2. **Load and Split Dataset**:
   ```python
   iris = load_iris()
   X = iris.data
   y = iris.target
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

3. **Define a Function to Train and Log Models**:
   ```python
   def train_log_and_register_model(model, model_name, experiment_name, model_registry_name):
       mlflow.set_experiment(experiment_name)
       with mlflow.start_run(run_name=model_name) as run:
           model.fit(X_train, y_train)
           predictions = model.predict(X_test)
           accuracy = accuracy_score(y_test, predictions)
           mlflow.log_param("model_type", model_name)
           mlflow.log_metric("accuracy", accuracy)
           mlflow.sklearn.log_model(model, artifact_path=model_name)
           model_uri = f"runs:/{run.info.run_id}/{model_name}"
           registered_model = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
           print(f"Model {model_name} registered as {model_registry_name} with version {registered_model.version}")
   ```

4. **Train and Log Models**:
   ```python
   logreg_model = LogisticRegression(max_iter=200)
   train_log_and_register_model(logreg_model, "Logistic_Regression", "LogReg_Experiment", "Logistic_Regression_Model")

   rf_model = RandomForestClassifier(n_estimators=100)
   train_log_and_register_model(rf_model, "Random_Forest", "RandomForest_Experiment", "Random_Forest_Model")

   svm_model = SVC(kernel='linear')
   train_log_and_register_model(svm_model, "SVM", "SVM_Experiment", "SVM_Model")
   ```
![image](https://github.com/user-attachments/assets/c3534a71-579e-44bd-ae85-bcb4414e2037)


## Step 2: Making Inferences with Registered Models

### Example: Custom PyFunc Model

1. **Define a Custom PyFunc Model**:
   ```python
   import mlflow.pyfunc
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler

   class CustomModel(mlflow.pyfunc.PythonModel):
       def __init__(self):
           self.model = None
       
       def fit(self, X, y):
           self.model = np.linalg.lstsq(X, y, rcond=None)[0]
       
       def predict(self, context, model_input):
           return np.dot(model_input, self.model)
   ```

2. **Train and Log the Custom Model**:
   ```python
   mlflow.set_experiment("CustomPyFuncExperiment")
   X = np.random.rand(100, 3)
   y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)

   with mlflow.start_run():
       custom_model = CustomModel()
       custom_model.fit(X_train_scaled, y_train)
       mlflow.pyfunc.log_model("custom_model", python_model=custom_model)
       run_id = mlflow.active_run().info.run_id
       print(f"Model logged with run_id: {run_id}")
   ```

3. **Load and Make Predictions**:
   ```python
   loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/custom_model")
   new_data = np.random.rand(10, 3)
   new_data_scaled = scaler.transform(new_data)
   predictions = loaded_model.predict(new_data_scaled)
   print("Predictions from loaded model:")
   print(predictions)
   ```

By following this guide, you can effectively log trained models using MLflow's Model Registry and make predictions with those models. This process enhances reproducibility and collaboration in machine learning projects.

# MLflow Model Deployment with Docker

This project demonstrates how to deploy machine learning models trained and logged with MLflow using Docker, and how to make inferences using these deployed models.

## Prerequisites

- Docker installed on your system
- MLflow
- Python 3.11 or later

## Project Structure

- `Dockerfile`: Defines the Docker image for deploying the models
- `mlflow_load_models.py`: Python script to load models and make predictions
- `mlruns/`: Directory containing MLflow model artifacts
- `requirements.txt`: List of Python dependencies

## Steps to Deploy and Run

1. **Build the Docker Image**

   Run the following command in the project root directory:

   ```
   docker build -t mlflow-model-deployment .
   ```

2. **Run the Docker Container**

   Execute the following command to run the container:

   ```
   docker run --add-host=host.docker.internal:host-gateway mlflow-model-deployment
   ```

   This command allows the container to access the MLflow tracking server running on your host machine.

3. **View the Results**

   The container will run the `mlflow_load_models.py` script, which loads the models from the MLflow Model Registry and makes predictions using the Iris dataset.

## How It Works

1. **Dockerfile**

   The Dockerfile:
   - Uses Python 3.11 slim image as the base
   - Sets up the working directory
   - Copies and installs the requirements
   - Copies the MLflow runs and the Python script
   - Sets the MLflow tracking URI
   - Runs the Python script

2. **mlflow_load_models.py**

   This script:
   - Connects to the MLflow tracking server
   - Loads the latest versions of specified models from the Model Registry
   - Prints details about each loaded model
   - Makes predictions using the Iris dataset
   - Displays the predictions

3. **Model Loading**

   The script loads three models:
   - Logistic Regression
   - Random Forest
   - SVM

   Each model is loaded using the MLflow API, and its details are printed.

4. **Making Predictions**

   The script uses the loaded models to make predictions on the Iris dataset and displays the results.

## Customization

- To deploy different models, modify the `models_to_load` list in `mlflow_load_models.py`.
- Adjust the `requirements.txt` file if your models require different dependencies.
- Modify the `mlflow_load_models.py` script to use your own dataset for predictions.

## Troubleshooting

- Ensure that your MLflow tracking server is running and accessible from the Docker container.
- If you encounter issues with model loading, check that the model names in `mlflow_load_models.py` match those in your MLflow Model Registry.
- Verify that all necessary model artifacts are present in the `mlruns/` directory.

## Conclusion

This setup allows you to easily deploy MLflow-logged models using Docker, making it simple to use these models for inference in various 
environments.

</br>

# Docker Build and Run
1. Update path of mlflow artifacts to point to docker workspace folder
    ```
    chmod +x mlflow_path_update.sh 
    ```

2. Run the shell script to chnage the paths

    ```
    ./mlflow_path_update.sh
    ```

3. Start and run mlflow server
   ```
   mlflow server --host 0.0.0.0
   ```

4. Build the docker image
   ```
   docker build -t mlflow-app .
   ```

5. Run the docker image
   ```
   docker run --add-host=host.docker.internal:host-gateway mlflow-app
   ```

6. Run the docker image in interactive mode
   ```
   docker run --add-host=host.docker.internal:host-gateway -it mlflow-app /bin/bash
   ```
7. Run with volume mount
   ```
   docker run --add-host=host.docker.internal:host-gateway -v $(pwd)/mlruns:/workspace/mlruns mlflow-app
   ```

