from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict
import mlflow
import mlflow.pytorch
import os
import logging


import sys
import os

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add mlflow directory to sys.path
mlflow_dir = os.path.join(project_root, "mlflow")
sys.path.append(mlflow_dir)

# Import MLflow tracking utilities
from tracking_uri import get_tracking_uri

# Setup logging
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "model_training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set MLflow tracking URI
mlflow.set_tracking_uri(get_tracking_uri())

# Ensure MLflow Experiment Exists
EXPERIMENT_NAME = "ABSA Sentiment Analysis"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(EXPERIMENT_NAME)

mlflow.set_experiment(EXPERIMENT_NAME)

# Define Dataset
class ABSADataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_len: int):
        self.sentences = data["Sentence"].values
        self.aspects = data["Aspect"].values
        self.labels = data["Sentiment"].apply(self.map_label).values
        self.tokenizer = tokenizer
        self.max_len = max_len

    @staticmethod
    def map_label(label: str) -> int:
        label_map = {"negative": 0, "neutral": 1, "positive": 2, "conflict": 3}
        return label_map.get(label, 1)  # Default to neutral if label is missing

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        aspect = self.aspects[index]
        label = self.labels[index]
        input_text = f"[ASPECT] {aspect} [SEP] {sentence}"
        encoding = self.tokenizer(
            input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Strategy Interface
class ABSAStrategy(ABC):
    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        pass
    
    @abstractmethod
    def analyze_sentiment(self, aspect: str, sentence: str) -> Dict[str, str]:
        pass

# BERT Strategy with MLflow
class BERTSentimentAnalysisStrategy(ABSAStrategy):
    def __init__(self, model_name: str = "bert-base-uncased", max_len: int = 128, batch_size: int = 16):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=1)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, epochs: int = 4, learning_rate: float = 5e-5):
        logging.info("Training started...")
        try:
            train_dataset = ABSADataset(train_data, self.tokenizer, self.max_len)
            self.val_dataset = ABSADataset(val_data, self.tokenizer, self.max_len)

            # Ensure the models directory exists
            model_dir = os.path.abspath("../models/transformers")
            os.makedirs(model_dir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=model_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_strategy="epoch",
                evaluation_strategy="epoch",
                save_strategy="epoch",
            )

            with mlflow.start_run():
                # Log Hyperparameters
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", self.batch_size)
                mlflow.log_param("learning_rate", learning_rate)

                self.trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=self.val_dataset,
                    compute_metrics=self.compute_metrics,
                )
                self.trainer.train()
                self.training_history = self.trainer.state.log_history

                # Save Model
                save_path = os.path.abspath("../models/saved_models")
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)

                # Log Model to MLflow
                mlflow.pytorch.log_model(self.model, "ABSA_model")
                print(f"✅ Model saved at {save_path}")
        except Exception as e:
            logging.error(f"Training failed: {e}")

    def analyze_sentiment(self, aspect: str, sentence: str) -> Dict[str, str]:
        if not aspect or not sentence:
            return {"error": "Aspect and sentence must be provided"}

        input_text = f"[ASPECT] {aspect} [SEP] {sentence}"
        encoding = self.tokenizer(
            input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        label_map = {0: "negative", 1: "neutral", 2: "positive", 3: "conflict"}

        return {"aspect": aspect, "sentiment": label_map.get(predicted_label, "unknown")}

# Context Class
class ABSAContext:
    def __init__(self, strategy: ABSAStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: ABSAStrategy):
        self.strategy = strategy
    
    def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        self.strategy.train(train_data, val_data)
    
    def predict_sentiment(self, aspect: str, sentence: str) -> Dict[str, str]:
        return self.strategy.analyze_sentiment(aspect, sentence)
















# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from torch.utils.data import Dataset
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import torch
# import pandas as pd
# from abc import ABC, abstractmethod
# from typing import Dict
# import mlflow
# import mlflow.pytorch
# import os
# import logging

# # Setup logging
# LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
# if not os.path.exists(LOG_DIR):
#     os.makedirs(LOG_DIR)

# logging.basicConfig(
#     filename=os.path.join(LOG_DIR, "model_training.log"),
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# # Set local MLflow tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# # Ensure MLflow Experiment Exists
# experiment_name = "ABSA Sentiment Analysis"
# experiment = mlflow.get_experiment_by_name(experiment_name)
# if experiment is None:
#     mlflow.create_experiment(experiment_name)

# mlflow.set_experiment(experiment_name)

# # Define Dataset
# class ABSADataset(Dataset):
#     def __init__(self, data: pd.DataFrame, tokenizer, max_len: int):
#         self.sentences = data["Sentence"].values
#         self.aspects = data["Aspect"].values
#         self.labels = data["Sentiment"].apply(self.map_label).values
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     @staticmethod
#     def map_label(label: str) -> int:
#         label_map = {"negative": 0, "neutral": 1, "positive": 2, "conflict": 3}
#         return label_map.get(label, 1)  # Default to neutral if label is missing

#     def __len__(self):
#         return len(self.sentences)

#     def __getitem__(self, index):
#         sentence = self.sentences[index]
#         aspect = self.aspects[index]
#         label = self.labels[index]
#         input_text = f"[ASPECT] {aspect} [SEP] {sentence}"
#         encoding = self.tokenizer(
#             input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
#         )
#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "label": torch.tensor(label, dtype=torch.long),
#         }

# # Strategy Interface
# class ABSAStrategy(ABC):
#     @abstractmethod
#     def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
#         pass
    
#     @abstractmethod
#     def analyze_sentiment(self, aspect: str, sentence: str) -> Dict[str, str]:
#         pass

# # BERT Strategy with MLflow
# class BERTSentimentAnalysisStrategy(ABSAStrategy):
#     def __init__(self, model_name: str = "bert-base-uncased", max_len: int = 128, batch_size: int = 16):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
#         self.max_len = max_len
#         self.batch_size = batch_size
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
    
#     def compute_metrics(self, pred):
#         labels = pred.label_ids
#         preds = pred.predictions.argmax(-1)
#         acc = accuracy_score(labels, preds)
#         precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=1)

#         # Log metrics to MLflow
#         mlflow.log_metric("accuracy", acc)
#         mlflow.log_metric("precision", precision)
#         mlflow.log_metric("recall", recall)
#         mlflow.log_metric("f1", f1)

#         return {
#             'accuracy': acc,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#         }
    
#     def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, epochs: int = 4, learning_rate: float = 5e-5):
#         logging.info("Training started...")
#         try:
#             train_dataset = ABSADataset(train_data, self.tokenizer, self.max_len)
#             self.val_dataset = ABSADataset(val_data, self.tokenizer, self.max_len)

#             # Ensure the models directory exists
#             model_dir = os.path.abspath("../models/transformers")
#             os.makedirs(model_dir, exist_ok=True)

#             training_args = TrainingArguments(
#                 output_dir=model_dir,
#                 num_train_epochs=epochs,
#                 per_device_train_batch_size=self.batch_size,
#                 per_device_eval_batch_size=self.batch_size,
#                 warmup_steps=500,
#                 weight_decay=0.01,
#                 logging_dir="./logs",
#                 logging_strategy="epoch",
#                 evaluation_strategy="epoch",
#                 save_strategy="epoch",
#             )

#             with mlflow.start_run():
#                 # Log Hyperparameters
#                 mlflow.log_param("epochs", epochs)
#                 mlflow.log_param("batch_size", self.batch_size)
#                 mlflow.log_param("learning_rate", learning_rate)

#                 self.trainer = Trainer(
#                     model=self.model,
#                     args=training_args,
#                     train_dataset=train_dataset,
#                     eval_dataset=self.val_dataset,
#                     compute_metrics=self.compute_metrics,
#                 )
#                 self.trainer.train()
#                 self.training_history = self.trainer.state.log_history

#                 # Save Model
#                 save_path = os.path.abspath("../models/saved_models")
#                 os.makedirs(save_path, exist_ok=True)
#                 self.model.save_pretrained(save_path)
#                 self.tokenizer.save_pretrained(save_path)

#                 # Log Model to MLflow
#                 mlflow.pytorch.log_model(self.model, "ABSA_model")
#                 print(f"✅ Model saved at {save_path}")
#         except Exception as e:
#             logging.error(f"Training failed: {e}")

#     def analyze_sentiment(self, aspect: str, sentence: str) -> Dict[str, str]:
#         if not aspect or not sentence:
#             return {"error": "Aspect and sentence must be provided"}

#         input_text = f"[ASPECT] {aspect} [SEP] {sentence}"
#         encoding = self.tokenizer(
#             input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
#         )
#         input_ids = encoding["input_ids"].to(self.device)
#         attention_mask = encoding["attention_mask"].to(self.device)

#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         predicted_label = torch.argmax(logits, dim=1).item()
#         label_map = {0: "negative", 1: "neutral", 2: "positive", 3: "conflict"}

#         return {"aspect": aspect, "sentiment": label_map.get(predicted_label, "unknown")}

# # Context Class
# class ABSAContext:
#     def __init__(self, strategy: ABSAStrategy):
#         self.strategy = strategy
    
#     def set_strategy(self, strategy: ABSAStrategy):
#         self.strategy = strategy
    
#     def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
#         self.strategy.train(train_data, val_data)
    
#     def predict_sentiment(self, aspect: str, sentence: str) -> Dict[str, str]:
#         return self.strategy.analyze_sentiment(aspect, sentence)

