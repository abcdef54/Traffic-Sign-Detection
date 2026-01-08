import time
from collections import defaultdict

class PredictionStabilizer:
    def __init__(self, decay: float = 0.8, expiry_time: float = 5.0):
        self.scores = defaultdict(lambda: defaultdict(float))
        self.last_seen = {} 
        self.decay = decay
        self.expiry_time = expiry_time
        

        self.next_cleanup_check = time.time() + expiry_time 

    def vote(self, object_id, new_class_name, conf: float) -> str:
        current_time = time.time()
        
        self.last_seen[object_id] = current_time

        for cls in self.scores[object_id]:
            self.scores[object_id][cls] *= self.decay

        self.scores[object_id][new_class_name] += conf
        
        if current_time > self.next_cleanup_check:
            self._cleanup_stale_ids(current_time)
            self.next_cleanup_check = current_time + self.expiry_time

        current_scores = self.scores[object_id]
        if not current_scores: return new_class_name
        return max(current_scores, key=current_scores.get)

    def _cleanup_stale_ids(self, current_time):
        stale_ids = [
            oid for oid, last_time in self.last_seen.items() 
            if (current_time - last_time) > self.expiry_time
        ]
        
        if stale_ids:
            for oid in stale_ids:
                del self.scores[oid]
                del self.last_seen[oid]