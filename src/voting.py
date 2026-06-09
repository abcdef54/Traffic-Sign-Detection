from cv2.gapi.ot import ObjectTrackerParams
import time
from collections import defaultdict

class PredictionStabilizer:
    def __init__(self, decay: float = 0.5, expiry_time: float = 3.0, override_threshold: int = 3, instant_override_conf: float = 0.85, sustained_override_conf: float = 0.55):
        """
        decay_per_second: How much of the score is retained after 1 second of absence.
        expiry_time: Seconds of invisibility before an object is completely forgotten.
        """
        self.scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.last_seen: dict[str, float] = {}  # Keeps track of when we last saw an ID
        self.decay = decay
        self.expiry_time = expiry_time
        self.override_threshold = override_threshold
        self.contradiction_counts = defaultdict(int)
        self.instant_override_conf = instant_override_conf
        self.sustained_override_conf = sustained_override_conf
        
        self.next_cleanup_check = time.time() + expiry_time 

    def vote(self, object_id, new_class_name, conf: float) -> str:
        current_time = time.time()


        # Override Logic
        # Force-overrides prediction history if a new label exhibits absolute certainty (>=90% conf)
        # or sustained disagreement, preventing the conf stablization from gate keep the correct prediction
        # as the camera approaches and the model sees objects clearly up close.
        historical_winner = None
        if object_id in self.scores and self.scores[object_id]:
            historical_winner = max(self.scores[object_id].items(), key=lambda x: x[1])[0]

        if historical_winner and new_class_name != historical_winner:
            
            if conf >= self.instant_override_conf:
                self.scores[object_id].clear()
                self.contradiction_counts[object_id] = 0
                
            elif conf >= self.sustained_override_conf:
                self.contradiction_counts[object_id] += 1
                if self.contradiction_counts[object_id] >= self.override_threshold:
                    self.scores[object_id].clear()
                    self.contradiction_counts[object_id] = 0
            else:
                pass
        else:
            self.contradiction_counts[object_id] = 0
        
        if object_id in self.last_seen:
            elapsed = current_time - self.last_seen[object_id]

            if elapsed > self.expiry_time:
                self.scores[object_id].clear()
            elif elapsed > 0:
                time_decay = self.decay ** elapsed
                for cls in self.scores[object_id]:
                    self.scores[object_id][cls] *= time_decay
        
        self.last_seen[object_id] = current_time

        self.scores[object_id][new_class_name] += conf

        if current_time > self.next_cleanup_check:
            self._cleanup_stale_ids(current_time)
            self.next_cleanup_check = current_time + self.expiry_time

        
        current_scores = self.scores[object_id]
        if not current_scores:
            return new_class_name, conf
        
        winner, winner_conf = max(current_scores.items(), key=lambda item: item[1])

        total_scores = sum(current_scores.values())
        normalized_conf = winner_conf / total_scores if total_scores > 0 else conf

        return winner, normalized_conf

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