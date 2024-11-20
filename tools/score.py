# needed for f1 scorer:
import numpy as np
from sklearn.metrics import f1_score

def multioutput_f1_score(y_true, y_pred):
    """
    Computes an average weighted F1 score across multiple output categories.

    Args:
        y_true: The "Ground truth" binary labels (2D array or DataFrame).
        y_pred: The "Predicted" binary labels (2D array or DataFrame).

    Returns:
        float: Average weighted F1 score across all output categories.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1_scores = [
        f1_score(y_true[:, i], y_pred[:, i], average='weighted', zero_division=1)
        for i in range(y_true.shape[1])
    ]

    return np.mean(f1_scores)