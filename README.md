# Advanced Sentiment Analysis of Product Reviews

![Screenshot 2025-03-16 000947](https://github.com/user-attachments/assets/28675adf-b087-4b65-9e17-548413637810)

---

A full-stack Aspect-Based Sentiment Analysis (ABSA) system that scrapes real user reviews and extracts opinions/sentiments from the reviews using NLP and deep learning techniques and then saves the data in a cloud database for later use to visualize and gain insights about the state of the business.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [How This Works](#how-this-works)
- [Screenshots](#screenshots)

## Features

- **Aspect-Based Sentiment Analysis (ABSA)** – Extracts sentiments on specific aspects in reviews.
- **Fine-tuned BERT Model** – Trained on custom datasets for aspect based sentiment analysis.
- **Enhanced aspect and sentiment understanding** - Used multiple review aspect understanding methods for high accuracy.
- **Schedulers and Scraping** – Scrapes customer reviews periodically.
- **MLflow Model Tracking** – Logs experiments for model performance comparisons.
- **Scalable Deployment** – Deployed on Google Cloud (GCP) with Kubernetes.
- **CI/CD with GitHub Actions** – Automated testing & deployment.

## Tech Stack

- Machine Learning: Python, Pytorch, BERT, SentenceTransformers, Spacy
- Backend: Flask, MLflow, MongoDB, Scrapy, APScheduler, Power BI
- Frontend: React.js
- Cloud & DevOps: Docker, Kubernetes, Google Cloud (GCP), GitHub Actions

## Installation

### Local Setup

1. Clone this repo:

```
git clone https://github.com/GeenethMedagedara/Advanced_Sentiment_Analysis_Restaurant_Reviews.git
cd sentiment-analysis-project
```

2. Build & run Docker containers:

```
docker-compose up --build
```

3. Access the frontend at:

```
http://localhost:8080
```

4. Change hyperparameters (Optional)

sentiment-analysis-project -> src -> models -> fine_tune.py

![Screenshot 2025-03-16 065632](https://github.com/user-attachments/assets/bb364ba2-2272-4282-bdf7-8225c4e9ffee)

5. Train model by running the modeling.ipynb

Mlflow docker container must be active !

![Screenshot 2025-03-16 070436](https://github.com/user-attachments/assets/0a7c93fa-cce0-443c-8dff-9ee771db630b)

6. Restart all the other docker containers

- Watch the predict pipeline run automatically and the sentiment predicted dataset will be saved in mongoDB
- Add custom inputs using the UI

7. Loaded the predicted dataset to Power BI for visualization

Within Power BI go to **Home -> Transform Data -> Python Script**
Enter the following code along with your credentials

```python
import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://username:password@your-cluster.mongodb.net/")
db = client["your_database"]
collection = db["predicted_reviews"]

# Load data into DataFrame
data = pd.DataFrame(list(collection.find()))

# Drop MongoDB's default `_id` field if needed
data.drop(columns=["_id"], inplace=True, errors="ignore")

# Output DataFrame to Power BI
data
```
Click OK → Load into Power BI

## How This Works

- **React frontend** acts as the UI for the user to input the custom reviews along with the user preferred aspect.
- **Flask backend** acts as the api that handles all the requests and connects with mlflow to load the model.
- **Mlflow** acts as a storage to store the trained models along with the evaluation metrics.
- **Predict Pipeline** does all the work. When activated, first the model will be loaded from **Mlflow**, the scheduled pipeline will not start unless the model is loaded. Then the **Scrapy** web scraper scrapes the user reviews periodically using a scheduler called **APScheduler**, and then the scraped data is stored within the files as a CSV. Then EDA.ipynb is called to perform basic EDA along with basic data cleaning. Then preprocessing.ipynb is called to perform more data cleaning and preprocessing before predicting the sentiments.  Then predict.ipynb is called to predict the sentiments. Then the dataset with the predicted sentiments will be stored in MongoDB. (Meanwhile all the logs will be saved in the logs directory)

---

> **BERT** is a transformer-based model designed for contextual word representations which understands words in relation to their surroundings, making it powerful for Aspect-Based Sentiment Analysis (ABSA). 
> **spaCy** is a lightweight NLP library used for efficient text preprocessing. I used this to tokenize the text and used the space word similarity to pinpoint the correct aspect names.
> **Sentence Transformers (SBERT)** generate dense vector embeddings that capture semantic similarity between sentences. I used this to convert reviews into embeddings to compare aspect sentiment across different reviews.
> Finally I used the spacy word embeddings and Sentence Transformers to detect the correct aspects and used the pre-trained BERT model to get the sentiment. This increased the ABSA accuracy of the project.

## Screenshots

Power BI

![Screenshot 2025-03-15 235922](https://github.com/user-attachments/assets/d1d478e7-74e6-43a9-ab2c-e63843f41d6e)

Mlflow model evaluation matrix

![Screenshot 2025-03-16 005933](https://github.com/user-attachments/assets/6f5269c7-777f-42a1-89c7-9e5f776ea610)

![Screenshot 2025-03-16 010004](https://github.com/user-attachments/assets/c102c540-27e6-4a30-b6ab-5b3b4c7bed11)

![Screenshot 2025-03-16 010026](https://github.com/user-attachments/assets/56176ac4-f92c-4452-a77c-8127894730bc)

Mongo DB database with saved dataset

![Screenshot 2025-03-16 112117](https://github.com/user-attachments/assets/95653943-d951-4fec-b45e-61b584e96601)
