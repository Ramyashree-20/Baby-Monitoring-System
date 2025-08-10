# image_to_timestamp_logger.py
import torch
import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
import requests  
from model import get_model
from dataset import REV_CLASS_MAPPING

class ImageDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(num_classes=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        print(f"[Image Logger] Model '{model_path}' loaded on {self.device}.")

    def get_state_from_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "unknown"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        highest_score_state = ('unknown', 0.0)
        for score, label in zip(outputs[0]['scores'], outputs[0]['labels']):
            if score > highest_score_state[1]:
                highest_score_state = (REV_CLASS_MAPPING.get(label.item(), "unknown"), score)
        return highest_score_state[0]

def start_logging():
    """
    Watches the folder for new images, logs them, and sends notifications for key events.
    """
    MODEL_PATH = "baby_state_model.pth"
    IMAGE_FOLDER = "new_images_to_log"
    LOG_FILE = "sleep_log.csv"
    CHECK_INTERVAL_SECONDS = 5

    detector = ImageDetector(MODEL_PATH)
    processed_files = set()
    last_notified_state = None # To avoid sending duplicate notifications

    if not os.path.exists(IMAGE_FOLDER): os.makedirs(IMAGE_FOLDER)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(['timestamp', 'detected_state'])

    print(f"[Image Logger] Watching folder: '{IMAGE_FOLDER}' for new images.")

    try:
        while True:
            current_image_files = {f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
            new_files = current_image_files - processed_files

            if new_files:
                print(f"[Image Logger] Found {len(new_files)} new image(s)...")
                for filename in sorted(list(new_files)):
                    image_path = os.path.join(IMAGE_FOLDER, filename)
                    detected_state = detector.get_state_from_image(image_path)
                    timestamp = datetime.now()

                    if detected_state != "unknown":
                        with open(LOG_FILE, 'a', newline='') as f:
                            csv.writer(f).writerow([timestamp.isoformat(), detected_state])
                        print(f"[Image Logger] Logged: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {filename} -> {detected_state}")

                    if detected_state in ['awake', 'cry'] and detected_state != last_notified_state:
                        message = f"Baby is now {detected_state}!"
                        print(f"[Image Logger] Sending notification: {message}")
                        try:
                            # Send the alert to the Flask app's endpoint
                            requests.post('http://127.0.0.1:5000/add_notification', json={'message': message})
                            last_notified_state = detected_state # Update last state to prevent spam
                        except requests.exceptions.RequestException as e:
                            print(f"[Image Logger] Could not connect to Flask app to send notification: {e}")
                    elif detected_state == 'sleep':
                        last_notified_state = 'sleep' # Reset when baby is sleeping

                    processed_files.add(filename)

            time.sleep(CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("[Image Logger] Stopping.")