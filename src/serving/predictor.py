# src/serving/predictor.py
import pandas as pd
import numpy as np
from typing import List

def build_features(records: List[dict], feature_names: List[str]) -> List[List[float]]:
    engineered = []
    for record in records:
        wind_spd = record["wind_speed"]
        wind_dir = record["wind_direction"]
        turb_int = record["turbulence_intensity"]
        wind_dir_rad = np.radians(wind_dir)

        features = {
            **record,
            "wind_direction_sin":  np.sin(wind_dir_rad),
            "wind_direction_cos":  np.cos(wind_dir_rad),
            "wind_speed_squared":  wind_spd ** 2,
            "wind_speed_cubed":    wind_spd ** 3,
            "wake_adjusted_wind":  wind_spd * (1 - turb_int),
        }
        try:
            engineered.append([features[col] for col in feature_names])
        except KeyError as e:
            raise ValueError(f"Missing required engineered feature: {e}")
    return pd.DataFrame(engineered, columns=feature_names)