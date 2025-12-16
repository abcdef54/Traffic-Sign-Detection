from collections import deque, defaultdict

# class PredictionStabilizer:
#     def __init__(self, history_length: int = 10) -> None:
#         """history_length: How many past frames to remember for voting"""
#         self.history = defaultdict(lambda: deque(maxlen=history_length))
    
#     def vote(self, object_id, new_class_name, confident: float) -> str:
#         """
#         Adds the new detection to history and returns the class with the 
#         highest cumulative confidence score.
#         """
#         # 1. Add new detection tuple (name, conf) to history
#         self.history[object_id].append((new_class_name, confident))

#         # 2. Calculate Weighted Scores
#         # Logic: Score = Sum of confidences
#         # Example: [('Stop', 0.8), ('Stop', 0.7), ('Yield', 0.9)]
#         # Stop Score: 1.5 | Yield Score: 0.9 -> Winner: Stop
#         scores = defaultdict(float)
        
#         for cls_name, conf in self.history[object_id]:
#             scores[cls_name] += conf

#         # 3. Find the winner (Key with the highest value)
#         winner = max(scores, key=scores.get)

#         return winner


from collections import deque, defaultdict

class PredictionStabilizer:
    def __init__(self, history_length: int = 15, stability_threshold: int = 5) -> None:
        """
        history_length: How many past frames to remember for the raw vote.
        stability_threshold: How many CONSECUTIVE times a new class must win 
                             the raw vote before we actually switch the label.
        """
        self.history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Stores the currently displayed label for each ID
        self.locked_label = {} 
        
        # Counts how many times a challenger has beaten the locked label
        self.streak_counter = defaultdict(int) 

        self.stability_threshold = stability_threshold

    def vote(self, object_id, new_class_name, confident: float) -> str:
        # 1. Update History
        self.history[object_id].append((new_class_name, confident))

        # 2. Calculate Raw Weighted Scores (Your original logic)
        scores = defaultdict(float)
        for cls_name, conf in self.history[object_id]:
            # Optional: Give more weight to recent frames
            # score = conf * (index / len) ... (keeping it simple for now)
            scores[cls_name] += conf

        # 3. Determine the "Challenger" (Who is mathematically winning right now?)
        challenger = max(scores, key=scores.get)

        # 4. Handle First-Time Detection
        if object_id not in self.locked_label:
            self.locked_label[object_id] = challenger
            self.streak_counter[object_id] = 0
            return challenger

        current_winner = self.locked_label[object_id]

        # 5. The "Stickiness" Logic
        if challenger == current_winner:
            # The challenger matches our current locked label. 
            # Reset streak (we are stable).
            self.streak_counter[object_id] = 0
            return current_winner
        else:
            # A new class is winning the vote!
            self.streak_counter[object_id] += 1
            
            # Only switch if they have maintained the lead for 'N' frames
            if self.streak_counter[object_id] >= self.stability_threshold:
                self.locked_label[object_id] = challenger
                self.streak_counter[object_id] = 0 # Reset streak for the new king
                return challenger
            else:
                # They haven't proven themselves yet. 
                # Return the OLD (locked) label to prevent flickering.
                return current_winner