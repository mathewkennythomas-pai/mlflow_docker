# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /workspace

# Copy the requirements.txt file to the working directory
COPY requirements.txt /workspace/

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the train.py script into the container's working directory
COPY mlflow_load_models.py /workspace/

# Copy the mlruns folder to the workspace
COPY mlruns /workspace/mlruns

# Optionally expose the MLflow server port
EXPOSE 5000

# Command to run the Python script when the container starts
CMD ["python", "mlflow_load_models.py"]

