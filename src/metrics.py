from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_rrs(r_ij: List[List[int]]) -> float:
    """
    r_ij: shape [N, 3], each entry in {0,1}
    RRS = (1/(3N)) * sum_i sum_j r_{i,j}  :contentReference[oaicite:7]{index=7}
    """
    arr = np.array(r_ij, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected r_ij shape [N,3], got {arr.shape}")
    return float(arr.mean())

def compute_acc_f1(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    return {"acc": acc, "f1": f1}
