# mlflow_docker
Deploying ML models using MLFlow and Docker

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
   docker run --add-host=host.docker.internal:host-gateway -v $(pwd)/model_artifacts:/workspace/mlruns mlflow-app
   ```

