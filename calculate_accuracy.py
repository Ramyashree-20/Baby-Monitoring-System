import torch
import cv2
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from model import get_model
from dataset import REV_CLASS_MAPPING, CLASS_MAPPING

def evaluate_model_performance(model_path, test_dir):
    """
    Loads a trained model and evaluates its performance on the test dataset.
    """
    # 1. Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4)  # 3 classes + background
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model '{model_path}' loaded on {device}.")

    # Lists to store the true labels and the model's predictions
    y_true = []
    y_pred = []

    test_images_path = os.path.join(test_dir, 'images')
    test_labels_path = os.path.join(test_dir, 'labels')

    print(f"Evaluating images in: {test_images_path}")

    # 2. Loop through every image in the test set
    for image_file in os.listdir(test_images_path):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # --- Get the True Label ---
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(test_labels_path, label_file)

        if not os.path.exists(label_path):
            print(f"Warning: Missing label file for {image_file}. Skipping.")
            continue

        with open(label_path, 'r') as f:
            line = f.readline().strip()
    
    # NEW: Check if the line is empty before processing
        if not line:
            print(f"Warning: Empty label file found for {image_file}. Skipping.")
            continue
    
    # If the line is not empty, proceed as before
        try:
            true_class_id = int(line.split()[0])
            y_true.append(true_class_id + 1)
        except (IndexError, ValueError):
            print(f"Warning: Malformed line in label file for {image_file}. Skipping.")
            continue

        # --- Get the Model's Prediction ---
        image_path = os.path.join(test_images_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(image_tensor)[0]

        # Find the prediction with the highest score
        if len(preds['scores']) > 0:
            best_score_index = torch.argmax(preds['scores'])
            predicted_class_id = preds['labels'][best_score_index].item()
            y_pred.append(predicted_class_id)
        else:
            y_pred.append(0)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n--- Model Performance Report ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n--- Confusion Matrix ---")
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Labels:", [REV_CLASS_MAPPING.get(l, "background") for l in labels])
    print(cm)


if __name__ == '__main__':
    evaluate_model_performance(
        model_path="baby_state_model.pth",
        test_dir="data/test"
    )