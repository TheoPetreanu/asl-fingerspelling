import cv2
import mediapipe as mp
import csv
import os
import time

# Add new signs here — letters or words
SIGNS = ['I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
SAMPLES_PER_SIGN = 150

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import urllib.request
if not os.path.exists("hand_landmarker.task"):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)

def get_landmarks_flat(hand):
    wrist = hand[0]
    # Normalise position relative to wrist
    coords = []
    for lm in hand:
        coords.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    
    # Normalise scale by dividing by the distance from wrist to middle finger MCP (landmark 9)
    # This makes the output the same regardless of hand distance from camera
    scale = max(abs(coords[9][0]), abs(coords[9][1]))
    if scale > 0:
        coords = [[v/scale for v in c] for c in coords]
    
    return [v for c in coords for v in c]

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    for sign in SIGNS:
        collected = 0
        print(f"\nGet ready for: {sign}")
        print("Press SPACE to start collecting...")

        # Wait for spacebar
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Sign: {sign} — press SPACE to start",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        # Collect samples
        while collected < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                flat = get_landmarks_flat(hand)

                # Save to CSV
                with open('training_data.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([sign] + flat)

                collected += 1
                cv2.putText(frame, f"{sign}: {collected}/{SAMPLES_PER_SIGN}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print("\nDone! training_data.csv saved.")