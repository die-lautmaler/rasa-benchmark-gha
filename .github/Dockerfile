# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && apt-get clean

# Install Rasa and any Python dependencies
RUN pip install -r benchmark/requirements.txt
RUN pip install rasa requests

# Define build-time arguments
ARG model_path=models/
ARG duckling_url=http://localhost:8000
ARG test_data_path=../../data/nlu.yml
ARG output_path=../../report/

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port on which Rasa will run
EXPOSE 5005

# Run the Rasa server as a background process and then run the Python script
CMD rasa run --enable-api --model ${model_path} --endpoints endpoints.yml --cors "*" & \
    sleep 300 && \
    python -m benchmark.bin check
