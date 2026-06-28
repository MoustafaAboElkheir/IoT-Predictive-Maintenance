"""
Predictive Maintenance — Model Definitions and Training Pipeline
================================================================
Defines all ML models used for failure prediction, including
hyperparameter grids and a unified training/evaluation interface.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score, classification_report)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

SENSOR_COLS = [
    "T24", "T30", "T50", "P15", "P30", "Nf", "Nc",
    "Ps30", "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32",
]
FEATURE_COLS = SENSOR_COLS + ["cycle", "setting_1", "setting_2"]

# Model registry with production-ready hyperparameters
MODEL_REGISTRY: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        scale_pos_weight=5,   # handles class imbalance
        random_state=42, eval_metric="logloss", verbosity=0,
    ),
}


def load_and_split(data_path: str, test_size: float = 0.30):
    """
    Load sensor data, apply 70/30 split, balance training set.
    Test set is left as-is (out-of-sample evaluation).
    """
    df = pd.read_csv(data_path)
    X, y = df[FEATURE_COLS], df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train class dist: {np.bincount(y_train)} | Test: {np.bincount(y_test)}")

    # Balance training set using undersampling (50-50)
    rus = RandomUnderSampler(random_state=42)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
    logger.info(f"Balanced train: {X_train_bal.shape} | {np.bincount(y_train_bal)}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_bal)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train_bal, y_test, scaler


def train_and_evaluate(model_name: str, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Train a single model and return comprehensive evaluation metrics."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[model_name]
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, proba), 4),
    }
    logger.info(f"  AUC-ROC={metrics['auc_roc']} | F1={metrics['f1']} | Recall={metrics['recall']}")
    return {"model": model, "metrics": metrics}


def run_all_models(data_path: str) -> Dict[str, Dict]:
    """Train and evaluate all models; return a comparison dictionary."""
    X_train, X_test, y_train, y_test, _ = load_and_split(data_path)
    results = {}
    for name in MODEL_REGISTRY:
        result = train_and_evaluate(name, X_train, y_train, X_test, y_test)
        results[name] = result["metrics"]
    return results
