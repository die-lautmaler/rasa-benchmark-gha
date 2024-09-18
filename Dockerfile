FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash
RUN /root/google-cloud-sdk/install.sh

# Add Google Cloud SDK to PATH
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

# Install additional Google Cloud components if needed
RUN gcloud components install beta

# Accept a build argument for the credentials JSON
ARG GOOGLE_CREDENTIALS_JSON

# Copy the credentials JSON into the container using the build argument
RUN echo "$GOOGLE_CREDENTIALS_JSON" > /root/credentials.json

# Authenticate with Google Cloud using the credentials JSON
RUN gcloud auth activate-service-account --key-file=/root/credentials.json


# Install Rasa and other Python dependencies
RUN python -m pip install rasa==3.6.20
RUN python -m pip install spacy==3.7.0
RUN pip install git+https://github.com/die-lautmaler/rasa_components.git
RUN pip install de-pipeline-lautmaler --extra-index-url=https://oauth2accesstoken:$GAR_TOKEN@europe-west3-python.pkg.dev/lautmaler-cca/cca-py/simple/

# Copy the benchmark directory from the action repository
COPY benchmark /action/benchmark

# Set the working directory
WORKDIR /github/workspace

# Install any needed packages specified in requirements.txt
COPY requirements.txt /action/requirements.txt
RUN pip install --no-cache-dir -r /action/requirements.txt

# Create an entrypoint script
COPY entrypoint.sh /action/entrypoint.sh
RUN chmod +x /action/entrypoint.sh

# Expose the Rasa server port
EXPOSE 5005

# Set the entrypoint
ENTRYPOINT ["/action/entrypoint.sh"]