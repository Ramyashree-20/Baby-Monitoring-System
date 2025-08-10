# train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BabyDataset
from model import get_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

TRAIN_DIR = 'data/train'
VALID_DIR = 'data/valid'
NUM_CLASSES = 4 
NUM_EPOCHS = 10
BATCH_SIZE = 4 
LEARNING_RATE = 0.005
IMG_WIDTH = 416 
IMG_HEIGHT = 416 

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    train_dataset = BabyDataset(TRAIN_DIR, IMG_WIDTH, IMG_HEIGHT)
    valid_dataset = BabyDataset(VALID_DIR, IMG_WIDTH, IMG_HEIGHT)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn 
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

   
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # Training phase
        model.train()
        train_loss_total = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, targets in progress_bar:
            # Move data to the correct device
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            
            # Sum all losses
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss_total += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")

        avg_train_loss = train_loss_total / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

    model_save_path = 'baby_state_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == '__main__':
    main()