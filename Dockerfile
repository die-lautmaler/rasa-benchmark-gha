FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rasa and other Python dependencies
RUN python -m pip install rasa==3.6.20

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