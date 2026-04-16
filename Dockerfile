FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "harness.py"]
