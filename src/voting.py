import time
from collections import defaultdict

class PredictionStabilizer:
    def __init__(self, decay: float = 0.8, expiry_time: float = 5.0):
        """
        decay: 0.8 (History retention)
        expiry_time: 5.0 seconds. If an object isn't seen for 5s, forget it.
        """
        self.scores = defaultdict(lambda: defaultdict(float))
        self.last_seen = {}  # Keeps track of when we last saw an ID
        self.decay = decay
        self.expiry_time = expiry_time
        
        # Throttling cleanup so we don't scan every single frame
        self.next_cleanup_check = time.time() + expiry_time 

    def vote(self, object_id, new_class_name, conf: float) -> str:
        current_time = time.time()
        
        # 1. Update Last Seen
        self.last_seen[object_id] = current_time

        # 2. Decay & Vote Logic
        # Decay ALL existing scores for this object
        for cls in self.scores[object_id]:
            self.scores[object_id][cls] *= self.decay

        self.scores[object_id][new_class_name] += conf
        
        # 3. AUTO-CLEANUP (The magic part)
        # Only run this check once every few seconds to save CPU
        if current_time > self.next_cleanup_check:
            self._cleanup_stale_ids(current_time)
            self.next_cleanup_check = current_time + self.expiry_time

        # 4. Return Winner
        current_scores = self.scores[object_id]
        if not current_scores: return new_class_name
        return max(current_scores, key=current_scores.get)

    def _cleanup_stale_ids(self, current_time):
        # Find IDs that haven't been seen in 'expiry_time' seconds
        # We use list() to create a copy of keys so we can delete while iterating
        stale_ids = [
            oid for oid, last_time in self.last_seen.items() 
            if (current_time - last_time) > self.expiry_time
        ]
        
        if stale_ids:
            # print(f"[Stabilizer] Cleaning up {len(stale_ids)} stale objects...")
            for oid in stale_ids:
                del self.scores[oid]
                del self.last_seen[oid]