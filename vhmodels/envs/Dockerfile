# Note: This docker file is used by the pkg's functions to create docker images for given models
# Note: If you need custom docker image, create new dockerfile

FROM mambaorg/micromamba:latest

# Set working directory
WORKDIR /app

# Copy project
COPY . .

USER root
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
USER mambauser

# Create environment
RUN micromamba create -y -f environment.yml && micromamba clean --all --yes

RUN micromamba run -n vhmodels-{project} pip install -e .

# Run the tests for the given model e.g. tests/test_dinobloom.py
CMD ["micromamba", "run", "-n", "vhmodels-{project}", "pytest", "tests/test_{project}.py"]
