# src/utils/metrics.py


###
### Functions that compute evaluation metrics
###


# imports
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# implementations

def calculate_accuracy(y_true: List, y_pred: List) -> float:
    """
    Calculates the accuracy score.
    """
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true: List, y_pred: List) -> float:
    """
    Calculates the weighted precision score for multi-class classification.
    """
    return precision_score(y_true, y_pred, average='weighted', zero_division=0)


def calculate_recall(y_true: List, y_pred: List) -> float:
    """
    Calculates the weighted recall score for multi-class classification.
    """
    return recall_score(y_true, y_pred, average='weighted', zero_division=0)


def calculate_f1_score(y_true: List, y_pred: List) -> float:
    """
    Calculates the weighted F1 score for multi-class classification.
    """
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

