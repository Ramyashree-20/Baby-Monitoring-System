#test_model.py
import torch
import cv2
import numpy as np
import os
import pygame
from model import get_model
from dataset import REV_CLASS_MAPPING

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=4)  # 3 classes + background
model.load_state_dict(torch.load("baby_state_model.pth", map_location=device))
model.to(device)
model.eval()

# Initialize pygame mixer for music
pygame.mixer.init()
pygame.mixer.music.load("soothing_music.mp3")

# Function to play music only if not already playing
def play_music():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # Loop infinitely

# Function to stop music
def stop_music():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Path to test images
test_folder = "data/test/images"
image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(test_folder, img_file)
    image = cv2.imread(img_path)
    orig = image.copy()

    # Preprocess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image)[0]

    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']

    cry_or_awake_detected = False

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.6:
            class_name = REV_CLASS_MAPPING[label.item()]
            if class_name in ['cry', 'awake']:
                cry_or_awake_detected = True

            # Draw bounding box
            box = box.int().cpu().numpy()
            cv2.rectangle(orig, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            text = f"{class_name}: {score:.2f}"
            cv2.putText(orig, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Music control
    if cry_or_awake_detected:
        play_music()
    else:
        stop_music()

    # Show result
    cv2.imshow("Detection", orig)
    key = cv2.waitKey(0)
    if key == 27:  # ESC key to break
        break

cv2.destroyAllWindows()
stop_music()
