# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && apt-get clean

# Install Rasa and any Python dependencies
RUN pip install python-dotenv
RUN pip install pyyaml
RUN pip install pandas
RUN pip install typer
RUN pip install google-cloud-dialogflow
RUN pip install google-api-python-client
RUN pip install gspread
RUN pip install gspread-dataframe
RUN pip install gspread-formatting
RUN pip install oauth2client
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install numpy
RUN python -m pip install rasa==3.6.20
RUN python -m pip install spacy==3.7.0


# Define build-time arguments
ARG model_path=models/
ARG duckling_url=http://localhost:8000
ARG test_data_path=../../data/nlu.yml
ARG output_path=../../report/

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port on which Rasa will run
EXPOSE 5005

# Create a startup script
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Run startup script when the container launches
CMD ["/app/startup.sh"]