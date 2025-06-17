FROM ultralytics/ultralytics:8.3.155

WORKDIR /workspace/
COPY . .
RUN pip install -e .
