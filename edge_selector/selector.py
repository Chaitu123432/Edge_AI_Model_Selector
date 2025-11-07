# edge_selector/selector.py
import json

class Selector:
    """
    Selects the best model based on predicted latency (from surrogate)
    and provided model attributes (accuracy, size, etc.)
    """

    def __init__(self, surrogate, sla_ms=200, weight_acc=0.7, weight_lat=0.3):
        self.surr = surrogate
        self.sla = sla_ms
        self.w_acc = weight_acc
        self.w_lat = weight_lat

    def select(self, context, actions):
        cores = context["cpu"]["cores_logical"]
        ram = context["ram"]["ram_total_mb"]

        candidates = []
        for a in actions:
            name = a.get("name")
            acc = a.get("accuracy", 0.8)

            try:
                pred = self.surr.predict([cores, ram])
            except Exception as e:
                print(f"⚠️ Prediction failed for {name}: {e}")
                continue

            if pred > self.sla:
                print(f"⚠️ Skipping {name}: Predicted latency {pred:.1f} > SLA {self.sla}")
                continue

            # Combined weighted score
            score = self.w_acc * acc - self.w_lat * (pred / self.sla)
            candidates.append((name, acc, pred, score))

        if not candidates:
            print("⚠️ No models fit the SLA! Returning best available anyway.")
            # fallback: choose minimal latency if all exceed SLA
            for a in actions:
                name = a.get("name")
                acc = a.get("accuracy", 0.8)
                pred = self.surr.predict([cores, ram])
                score = self.w_acc * acc - self.w_lat * (pred / self.sla)
                candidates.append((name, acc, pred, score))

        # pick highest-scoring model
        best = max(candidates, key=lambda x: x[3])
        return {
            "best_model": best[0],
            "pred_latency_ms": best[2],
            "accuracy": best[1],
            "score": best[3],
            "candidates": candidates
        }
