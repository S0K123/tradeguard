# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables
ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
ENV HF_TOKEN=""

# Run inference.py when the container launches
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
