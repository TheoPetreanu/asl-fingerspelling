import cv2
import mediapipe as mp
import pickle
import numpy as np
import threading
import time
import queue
import requests
import pygame
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("ELEVENLABS_API_KEY")

pygame.mixer.init()

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George voice

tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "text": text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
            }
        )
        pygame.mixer.music.unload()  # release file lock before overwriting
        with open("tts_output.mp3", "wb") as f:
            f.write(response.content)
        pygame.mixer.music.load("tts_output.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    tts_queue.put(text)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)

def get_landmarks_flat(hand):
    wrist = hand[0]
    coords = []
    for lm in hand:
        coords.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    scale = max(abs(coords[9][0]), abs(coords[9][1])) # scales relative to middle finger MCP
    if scale > 0:
        coords = [[v/scale for v in c] for c in coords]
    return [v for c in coords for v in c]

current_word = ""
current_prediction = ""
prediction_hold_start = 0
last_added_letter = ""
last_added_time = 0

HOLD_DURATION = 0.8
LETTER_COOLDOWN = 0.5

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        now = time.time()

        if result.hand_landmarks:
            hand = result.hand_landmarks[0] #hand is now a list of 21 landmarks
            flat = get_landmarks_flat(hand) # flat is a list of 63 numbers (coordinates for each landmark)
            prediction = clf.predict([flat])[0] # clf.predict() returns a letter
            confidence = clf.predict_proba([flat]).max()

            if confidence > 0.6:
                cv2.putText(frame, f"{prediction} ({confidence:.0%})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

                if prediction != current_prediction:
                    current_prediction = prediction
                    prediction_hold_start = now

                held = now - prediction_hold_start
                progress = min(held / HOLD_DURATION, 1.0)
                bar_x, bar_y = 10, 90
                bar_w, bar_h = 300, 12
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1) #(frame to draw on, top left corner, bottom right corner, colour in BGR, -1 means filled), draws a dark grey background rectangle
                fill_w = int(bar_w * progress) #how many pixels wide the fill should be, bar width * progress
                g = int(150 + 105 * progress) 
                b = int(255 - 255 * progress) # colour changes as progress changes
                if fill_w > 0:
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (b, g, 0), -1) # draws the fill
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1) # draws the outline (thickness 1)

                if held >= HOLD_DURATION:
                    if prediction != last_added_letter or (now - last_added_time) > LETTER_COOLDOWN: 
                        current_word += prediction
                        last_added_letter = prediction
                        last_added_time = now
                        prediction_hold_start = now
            else: # if confidence is < 0.6
                current_prediction = ""
                prediction_hold_start = 0
                cv2.putText(frame, "?", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        else: # if no hand detected
            current_prediction = ""
            prediction_hold_start = 0

        cv2.putText(frame, f"Word: {current_word}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=speak | BKSP=delete | Q=quit",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Sign Language", frame) # displays the frame in a windows called 'Sign Language'

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(" "):
            if current_word:
                speak(current_word)
                current_word = ""
                last_added_letter = ""
        elif key == ord("\b"): # backspace
            current_word = current_word[:-1]

cap.release()
cv2.destroyAllWindows()
