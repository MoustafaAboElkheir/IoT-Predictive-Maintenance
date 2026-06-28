"""Training pipeline for predictive maintenance."""
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler

SENSOR_COLS = ['T24','T30','T50','P15','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32']
FEATURE_COLS = SENSOR_COLS + ['cycle','setting_1','setting_2']

def train(data_path, model_name='xgboost'):
    df = pd.read_csv(data_path)
    X, y = df[FEATURE_COLS], df['failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    rus = RandomUnderSampler(random_state=42)
    X_train_b, y_train_b = rus.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_b)
    X_test_s  = scaler.transform(X_test)
    model = XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0) \
            if model_name == 'xgboost' else RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_s, y_train_b)
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:,1]
    print(f"AUC-ROC: {roc_auc_score(y_test, proba):.4f}")
    print(classification_report(y_test, preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/turbofan_sensors.csv')
    parser.add_argument('--model', default='xgboost')
    args = parser.parse_args()
    train(args.data, args.model)
