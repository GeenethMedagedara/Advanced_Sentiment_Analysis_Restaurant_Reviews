# Advanced Sentiment Analysis of Product Reviews

![Screenshot 2025-03-16 000947](https://github.com/user-attachments/assets/28675adf-b087-4b65-9e17-548413637810)

---

A full-stack Aspect-Based Sentiment Analysis (ABSA) system that scrapes real user reviews and extracts opinions/sentiments from the reviews using NLP and deep learning technics.

## Table of Contents

- [Features](https://github.com/GeenethMedagedara/Advanced_Sentiment_Analysis_Restaurant_Reviews/blob/master/README.md#features)
- [Tech Stack](https://github.com/GeenethMedagedara/Advanced_Sentiment_Analysis_Restaurant_Reviews/blob/master/README.md#tech-stack)
- [Installation](https://github.com/GeenethMedagedara/Advanced_Sentiment_Analysis_Restaurant_Reviews/blob/master/README.md#installation)
- [How This Works](https://github.com/GeenethMedagedara/Advanced_Sentiment_Analysis_Restaurant_Reviews/blob/master/README.md#how-this-works)
- [Screenshots](https://github.com/GeenethMedagedara/Advanced_Sentiment_Analysis_Restaurant_Reviews/blob/master/README.md#screenshots)

## Features

- **Aspect-Based Sentiment Analysis (ABSA)** – Extracts sentiments on specific topics in reviews.
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

4. Loaded the predicted dataset to Power BI for visualization

Within Power BI go to **Home -> Transform Data -> Python Script**
Enter the following code

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


## Screenshots

Power BI

![Screenshot 2025-03-15 235922](https://github.com/user-attachments/assets/d1d478e7-74e6-43a9-ab2c-e63843f41d6e)

Mlflow model evaluation matrix

![Screenshot 2025-03-16 005933](https://github.com/user-attachments/assets/6f5269c7-777f-42a1-89c7-9e5f776ea610)

![Screenshot 2025-03-16 010004](https://github.com/user-attachments/assets/c102c540-27e6-4a30-b6ab-5b3b4c7bed11)

![Screenshot 2025-03-16 010026](https://github.com/user-attachments/assets/56176ac4-f92c-4452-a77c-8127894730bc)
