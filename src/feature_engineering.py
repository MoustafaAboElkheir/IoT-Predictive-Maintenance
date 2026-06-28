"""Feature engineering for IoT sensor data — rolling statistics and lag features."""
import pandas as pd
import numpy as np

def add_rolling_features(df, sensor_cols, windows=[5, 10, 20]):
    for col in sensor_cols:
        for w in windows:
            df[f'{col}_rolling_mean_{w}'] = df.groupby('engine_id')[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean())
            df[f'{col}_rolling_std_{w}'] = df.groupby('engine_id')[col].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    return df

def add_lag_features(df, sensor_cols, lags=[1, 3, 5]):
    for col in sensor_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby('engine_id')[col].shift(lag).fillna(0)
    return df
