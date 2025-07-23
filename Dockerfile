FROM python:3.10-slim as python-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----
# Create stage for Poetry installation
FROM python-base as requirements-builder

# Creating a virtual environment just for poetry and install it with pip
ENV POETRY_VERSION=1.8.2
ENV PATH=/root/.local/bin:${PATH}
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false

COPY poetry.lock pyproject.toml /tmp/
RUN cd /tmp && poetry export --without-hashes -o /tmp/requirements.txt

# -----
# Create a new stage from the base python image
FROM python-base

# Copy requirements.txt exported from Poetry to app image
COPY --from=requirements-builder /tmp/requirements.txt /tmp/
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir --disable-pip-version-check -r /tmp/requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the rest of the project
COPY . .

ENV PYTHONPATH=/app:/app/src:$PYTHONPATH

# Expose ports for API and Jupyter
EXPOSE 8000
EXPOSE 8888

# Run FastAPI and Jupyter Lab
CMD ["bash", "-c", "uvicorn --app-dir src/ apps.main:app --host 0.0.0.0 --port 8000 & jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"]
