# monitor_and_play_music.py
import torch
import cv2
import os
import numpy as np
import time
import pygame  

from model import get_model
from dataset import REV_CLASS_MAPPING

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=4)  # 3 classes + background
model.load_state_dict(torch.load("baby_state_model.pth", map_location=device))
model.to(device).eval()

cap = cv2.VideoCapture(0)  

# Initialize music player
pygame.mixer.init()
pygame.mixer.music.load("soothing_music.mp3") 

def play_music():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # Loop forever

def stop_music():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

print("Monitoring started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    detected_states = []

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.6:
            x1, y1, x2, y2 = box.astype(int)
            class_name = REV_CLASS_MAPPING.get(label, "unknown")
            detected_states.append(class_name)

            # Draw
            color = (0, 255, 0)
            cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
            cv2.putText(orig, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Control music based on detections
    if 'awake' in detected_states or 'cry' in detected_states:
        play_music()
    else:
        stop_music()

    cv2.imshow("Baby Monitor", orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.mixer.quit()
