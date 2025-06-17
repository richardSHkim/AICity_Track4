FROM ultralytics/ultralytics:8.3.155

COPY src scripts configs README.md Dockerfile Dockerfile.jetson pyproject.toml uv.lock /workspace
WORKDIR /workspace
RUN pip install -e .
