FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

COPY mlruns /workspace/mlruns

COPY mlflow_load_models.py /workspace/

ENV MLFLOW_TRACKING_URI file://app/mlruns

CMD ["python", "mlflow_load_models.py"]