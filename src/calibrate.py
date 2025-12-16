# calibrate.py
import numpy as np
import cv2
import glob

# --- CONFIGURATION ---
CHESSBOARD_SIZE = (9, 6)  # Number of INTERNAL corners (width, height)
SQUARE_SIZE = 0.024       # Size of one square in meters (measure your printed paper!)
MIN_SAMPLES = 3          # How many good images we want before calculating

# ---------------------

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# This represents the "Real World" coordinates of the corners
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(0)

print(f"--- CAMERA CALIBRATION ---")
print(f"1. Hold the chessboard in front of the camera.")
print(f"2. Press 's' to save a valid frame (Need {MIN_SAMPLES} samples).")
print(f"3. Move the board to different angles/distances between shots.")
print(f"4. Press 'q' to quit.")

samples_taken = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # Draw the corners if found (Visual feedback)
    display_frame = frame.copy()
    if ret_corners:
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
        cv2.putText(display_frame, "PATTERN FOUND! Press 's' to save", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Show Chessboard...", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(display_frame, f"Samples: {samples_taken}/{MIN_SAMPLES}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Calibration', display_frame)

    key = cv2.waitKey(1) & 0xFF

    # 's' to Save a sample
    if key == ord('s') and ret_corners:
        objpoints.append(objp)
        
        # Refine corner locations for higher accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        samples_taken += 1
        print(f"Saved sample {samples_taken}/{MIN_SAMPLES}")

    # 'c' to Calibrate (if enough samples)
    if samples_taken >= MIN_SAMPLES:
        print("\nCalculating Camera Parameters... (This might take a moment)")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("\n============================================")
        print("CALIBRATION SUCCESSFUL!")
        print("============================================")
        print(f"Focal Length X (fx): {mtx[0,0]:.2f}")
        print(f"Focal Length Y (fy): {mtx[1,1]:.2f}")
        print(f"Optical Center (cx, cy): ({mtx[0,2]:.2f}, {mtx[1,2]:.2f})")
        print("--------------------------------------------")
        print(f"COPY THIS INTO src/distance.py:")
        print(f"focal_length = { (mtx[0,0] + mtx[1,1]) / 2 :.2f} ")
        print("============================================")
        break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()