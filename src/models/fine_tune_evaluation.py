from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

# Strategy Interface
class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, trainer, val_dataset):
        pass

# Concrete Strategy: Plot Training Curves
class TrainingCurvesStrategy(EvaluationStrategy):
    def evaluate(self, trainer, val_dataset):
        training_history = trainer.state.log_history
        plt.figure(figsize=(12, 4))

        # Loss curve
        plt.subplot(1, 2, 1)
        train_loss = [log['loss'] for log in training_history if 'loss' in log]
        val_loss = [log['eval_loss'] for log in training_history if 'eval_loss' in log]
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy curve
        plt.subplot(1, 2, 2)
        val_accuracy = [log['eval_accuracy'] for log in training_history if 'eval_accuracy' in log]
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

# Concrete Strategy: Precision, Recall, F1-score
class MetricsStrategy(EvaluationStrategy):
    def evaluate(self, trainer, val_dataset):
        predictions = trainer.predict(val_dataset)
        labels = predictions.label_ids
        preds = predictions.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

# Concrete Strategy: Confusion Matrix
class ConfusionMatrixStrategy(EvaluationStrategy):
    def evaluate(self, trainer, val_dataset):
        predictions = trainer.predict(val_dataset)
        labels = predictions.label_ids
        preds = predictions.predictions.argmax(-1)

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

# Concrete Strategy: ROC AUC Curve
class ROCAUCStrategy(EvaluationStrategy):
    def evaluate(self, trainer, val_dataset):
        predictions = trainer.predict(val_dataset)
        labels = predictions.label_ids
        preds = predictions.predictions

        labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])  # Assuming 4 classes
        n_classes = labels_bin.shape[1]

        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr['micro'], tpr['micro'], _ = roc_curve(labels_bin.ravel(), preds.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot(fpr['micro'], tpr['micro'], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC AUC Curve')
        plt.legend(loc='lower right')
        plt.show()

# Concrete Strategy: t-SNE Visualization
class TSNEStrategy(EvaluationStrategy):
    def evaluate(self, trainer, val_dataset):
        predictions = trainer.predict(val_dataset)
        labels = predictions.label_ids
        embeddings = predictions.predictions

        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette='viridis')
        plt.title('t-SNE Visualization of BERT Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()

# Context Class
class EvaluationContext:
    def __init__(self, strategy: EvaluationStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: EvaluationStrategy):
        self.strategy = strategy

    def execute_evaluation(self, trainer, val_dataset):
        self.strategy.evaluate(trainer, val_dataset)