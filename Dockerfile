FROM python:3.11-slim

WORKDIR /workspace

# Install uv
RUN pip install uv

# Copy pyproject.toml
COPY pyproject.toml /workspace/pyproject.toml

# Create a virtual environment and install dependencies using uv
# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock,from=pyproject.toml \
    uv pip sync --frozen --no-install-project --no-dev

# COPY requirements.txt /workspace/requirements.txt
# RUN pip install -r requirements.txt

COPY mlruns /workspace/mlruns

COPY mlflow_load_models.py /workspace/

ENV MLFLOW_TRACKING_URI file://app/mlruns

CMD ["python", "mlflow_load_models.py"]