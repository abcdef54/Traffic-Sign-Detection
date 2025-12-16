class DistanceEstimator:
    def __init__(self, focal_length: float = 800, real_height: float = 0.6, alpha: float = 0.2) -> None:
        """
        focal_length: Calibrated focal length (you might need to tweak this).
        real_height: The physical height of a traffic sign (~60cm = 0.6m).
        alpha: Smoothing factor (0.2 = keep 80% old value, 20% new value) - Avoid jittering.
        """
        self.FOCAL_LENGTH = focal_length
        self.REAL_HEIGHT = real_height
        self.ALPHA = alpha

        # Memory to store the previous smoothed distance for each object ID
        self.history = {}
    
    def calculate_distance(self, object_id, box_height_pixel) -> float:
        """"Return the smoothed distance in meters"""
        if box_height_pixel == 0.0: return 0

        # formula: Distance = (Real_height * focal_length) / image_height_in_pixels
        raw_dist = (self.REAL_HEIGHT * self.FOCAL_LENGTH) / box_height_pixel

        # If this is a new object (newly detected sign), trust the raw value 100%, since there are not memory to check
        if object_id not in self.history:
            self.history[object_id] = raw_dist
            return raw_dist
        
        prev_dist = self.history[object_id]
        # Formula: New = (Current * alpha) + (Old * (1 - alpha))
        smoothed_dist = (raw_dist * self.ALPHA) + (prev_dist * (1 - self.ALPHA))

        self.history[object_id] = smoothed_dist
        
        return smoothed_dist

