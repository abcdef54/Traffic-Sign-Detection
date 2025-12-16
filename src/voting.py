# File: src/stabilizer.py
from collections import defaultdict

class PredictionStabilizer:
    def __init__(self, decay: float = 0.8):
        """
        decay: 0.8 means history fades by 20% every frame. 
               Higher (0.95) = More stable, slower to react.
               Lower (0.6) = Reacts fast, but flickers more.
        """
        # {object_id: {'No Left Turn': 3.5, 'No Right Turn': 0.2}}
        self.scores = defaultdict(lambda: defaultdict(float))
        self.decay = decay

    def vote(self, object_id, new_class_name, conf: float) -> str:
        # Decay ALL existing scores for this object (History fades)
        for cls in self.scores[object_id]:
            self.scores[object_id][cls] *= self.decay

        self.scores[object_id][new_class_name] += conf

        current_scores = self.scores[object_id]
        if not current_scores: return new_class_name
        
        winner = max(current_scores, key=current_scores.get)
        return winner