import json
class Selector:
    def __init__(self, surrogate, sla_ms=200):
        self.surr = surrogate
        self.sla = sla_ms

    def select(self, context, actions):
        best=None; best_score=-9999
        for a in actions:
            pred = self.surr.predict([context['cpu']['cores_logical'],context['ram']['ram_total_mb']])
            if pred>self.sla: continue
            score = a.get('accuracy',0.8) - 0.01*pred
            if score>best_score:
                best_score=score; best=a
        return {'best':best,'score':best_score}