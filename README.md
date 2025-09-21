# Hotel Reservation Prediction

An end-to-end MLOps pipeline built using Google Cloud Platform (GCP) and Flask to predict whether a hotel reservation will be cancelled or honored.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Configuration](#configuration)  
- [Usage](#usage)  
  - [Training & Pipeline](#training--pipeline)  
  - [Model Serving](#model-serving)  
- [CI / CD / MLOps](#ci--cd--mlops)  
- [Technologies Used](#technologies-used)  

---

## Overview

This repository implements a machine learning pipeline that ingests hotel reservation data, trains a model to predict whether a booking will be cancelled or honored, and deploys the model via a REST API using Flask. The infrastructure and workflows leverage Google Cloud Platform for scalable and reproducible operations.

---

## Features

- Data preprocessing & feature engineering  
- Model training & evaluation  
- Automated ML pipeline (on GCP)  
- Dockerized Flask API for inference  
- CI/CD / orchestration for reproducible and scalable deployment  

---

## Project Structure

```text
├── artifacts/             # Model artifacts, logs, output from training
├── config/                # Configuration files (e.g. for GCP, credentials, hyperparameters)
├── custom_jenkins/        # Jenkins pipeline definitions or scripts, if applicable
├── notebooks/             # Exploratory data analysis, experimentation
├── pipeline/              # Pipeline code (data ingestion, transformation, training, etc.)
├── src/                   # Source code modules (data, models, utils)
├── templates/             # Flask API templates or HTML templates, if any
├── utils/                 # Utility scripts and helpers
├── Dockerfile             # For creating container to serve model
├── Jenkinsfile            # CI/CD pipeline script
├── application.py         # Entry point for serving predictions (Flask app)
├── requirements.txt       # Python package dependencies
└── setup.py               # Package installation script
```
## Getting Started
### Prerequisites

- Python 3.7+
- Google Cloud account with required services enabled (e.g. GCS, AI Platform / Vertex AI, Cloud Build, etc.)
- Docker (for containerization)
- Jenkins (if using the Jenkins pipeline)
- (Optional) Access to dataset of hotel reservations

### Installation

- Clone this repository:
```
git clone https://github.com/Sumit-Prasad01/Hotel-Reservation-Prediction.git
cd Hotel-Reservation-Prediction
```

### Create & activate a virtual environment (optional but recommended):
```
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies:
```
pip install -r requirements.txt
```

### If packaging:

- python setup.py install

### Configuration

- Copy or create configuration file(s) under config/ for access credentials to GCP services, storage buckets, etc.

- Set environment variables (if any) for API keys, database URIs, etc.

- For local testing, consider setting up a .env file and loading using a library like python-dotenv.

### Usage
- Training & Pipeline

- Use the notebooks in notebooks/ for exploratory data analysis.

- Use the scripts / pipeline folder to run full training end-to-end locally or on GCP.

### Example command:
```
python src/train.py --config config/train_config.yaml
```

- Tensorboard or logging tools can be used to monitor training.

- Model Serving

### To serve predictions, use application.py (Flask app).
```
python application.py
```

### Or start via Docker:
```
docker build -t hotel-reservation-predictor .
docker run -p 5000:5000 hotel-reservation-predictor
```

### API endpoints (examples):

- POST /predict — send input features, get prediction (canceled / honored)

- GET /health — check server health

### CI / CD / MLOps

- Jenkinsfile is included for automated pipeline runs: linting, training, test, build, deployment.

- Pipeline folder may contain workflow definitions to run on GCP or cloud build.

- Artifacts (models, logs) are stored in artifacts/ or GCP storage.

### Technologies Used

- Languages / Frameworks: Python, Flask

- Machine Learning: scikit-learn or similar (modify as per actual model)

- Cloud / MLOps: Google Cloud Platform services (e.g. GCS, AI Platform / Vertex AI, Cloud Build)

- CI/CD: Jenkins

- Containerization: Docker
