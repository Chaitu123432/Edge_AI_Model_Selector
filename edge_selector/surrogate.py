import json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Surrogate:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)

    def load(self, ndjson):
        rows = [json.loads(x) for x in open(ndjson)]
        X = [[r['device']['cpu']['cores_logical'], r['device']['ram']['ram_total_mb']] for r in rows]
        y = [r['latency_ms'] for r in rows]
        return np.array(X), np.array(y)

    def train(self, ndjson):
        X,y = self.load(ndjson)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2)
        self.model.fit(Xtr,ytr)
        print('RMSE', mean_squared_error(yte,self.model.predict(Xte),squared=False))

    def predict(self, context):
        return self.model.predict([context])[0]