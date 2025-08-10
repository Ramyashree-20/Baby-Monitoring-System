# camera_monitor.py
import torch
import cv2
import numpy as np
import time
import pygame
import requests
from model import get_model
from dataset import REV_CLASS_MAPPING

def play_music():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_music():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def start_monitoring(flask_app):
    """ The main monitoring loop. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=4)
    model.load_state_dict(torch.load("baby_state_model.pth", map_location=device))
    model.to(device).eval()

    pygame.mixer.init()
    pygame.mixer.music.load("soothing_music.mp3")

    cap = cv2.VideoCapture(0)
    print("[Camera Monitor] Started.")

    last_alert_state = None
    ALERT_COOLDOWN = 10 # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)

        detected_states = set()
        for score, label in zip(outputs[0]['scores'], outputs[0]['labels']):
            if score > 0.7: # Higher confidence for alerts
                detected_states.add(REV_CLASS_MAPPING.get(label.item()))

        current_state = "sleep" # Default state
        if 'cry' in detected_states:
            current_state = 'cry'
        elif 'awake' in detected_states:
            current_state = 'awake'

        # --- Music and Alert Logic ---
        if current_state in ['cry', 'awake']:
            play_music()
            if current_state != last_alert_state:
                message = f"Baby is {current_state}!"
                print(f"[Camera Monitor] Sending alert: {message}")
                try:
                    # Send alert to the Flask app
                    requests.post('http://127.0.0.1:5000/add_notification', json={'message': message})
                except requests.exceptions.RequestException as e:
                    print(f"[Camera Monitor] Could not connect to Flask app: {e}")
                last_alert_state = current_state
                time.sleep(ALERT_COOLDOWN) # Wait before sending another alert
        else:
            stop_music()
            last_alert_state = 'sleep'

        time.sleep(1) # Check once per second

    cap.release()
    pygame.mixer.quit()