# Base image
FROM python:3.9-slim

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