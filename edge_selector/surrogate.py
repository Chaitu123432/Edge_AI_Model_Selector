# edge_selector/surrogate.py
import os
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Surrogate:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)

    def load(self, ndjson_path):
        if not os.path.exists(ndjson_path):
            raise FileNotFoundError(f"{ndjson_path} not found")

        rows = []
        with open(ndjson_path, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    # Skip bad json lines but inform user
                    print(f"⚠️ Skipping bad JSON on line {i}: {e}")
                    continue

        if not rows:
            raise ValueError("No valid rows found in ndjson file")

        X = []
        y = []
        for r in rows:
            # defensive extraction of device features and latency
            try:
                cpu = r.get('device', {}).get('cpu', {})
                ram = r.get('device', {}).get('ram', {})
                cores_logical = cpu.get('cores_logical') or cpu.get('logical_cores') or cpu.get('cores') or 0
                ram_total_mb = ram.get('ram_total_mb') or ram.get('total_mb') or 0
                latency = r.get('latency_ms')
                if latency is None:
                    # maybe nested under other keys or use run-level latency
                    # skip row if latency missing
                    print("⚠️ Skipping row with missing latency.")
                    continue

                # Ensure numeric
                cores_logical = float(cores_logical)
                ram_total_mb = float(ram_total_mb)
                latency = float(latency)

                X.append([cores_logical, ram_total_mb])
                y.append(latency)
            except Exception as e:
                print(f"⚠️ Skipping row due to parsing error: {e}")
                continue

        if len(X) == 0:
            raise ValueError("No usable rows after parsing ndjson")

        return np.array(X), np.array(y)

    def train(self, ndjson_path):
        X, y = self.load(ndjson_path)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"🔢 Training data shape: Xtr={Xtr.shape}, Xte={Xte.shape}")
        self.model.fit(Xtr, ytr)

        preds = self.model.predict(Xte)
        mse = mean_squared_error(yte, preds)
        rmse = float(np.sqrt(mse))
        print('RMSE:', rmse)
        return rmse

    def predict(self, context):
        """Context should be an iterable/list of features in same order as training X:
           [cores_logical, ram_total_mb]
        """
        return float(self.model.predict([context])[0])
