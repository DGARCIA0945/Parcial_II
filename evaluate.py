import torch
import numpy as np
from sklearn.metrics import (accuracy_score, recall_score,
                             f1_score, mean_absolute_error, mean_squared_error)

def evaluate(model, loader, device='cuda'):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = torch.softmax(model(X_batch.to(device)), dim=-1)[:,1].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return (f1_score(y_true, y_pred, average='macro', zero_division=0),
            accuracy_score(y_true, y_pred),
            recall_score(y_true, y_pred, average='macro', zero_division=0),
            mean_absolute_error(y_true, y_prob),
            mean_squared_error(y_true, y_prob),
            y_prob, y_true)
