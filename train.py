import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import HiraganaDataset
from model import HiraganaClassifier
from tqdm import tqdm
import os

train_transform = transforms.Compose([
    transforms.RandomAffine(5, translate=(0.05, 0.05), shear=10, scale=(0.8,1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = HiraganaDataset("Hiragana_Split/train", transform=train_transform)
val_dataset = HiraganaDataset("Hiragana_Split/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

device = "cpu"

model = HiraganaClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
os.makedirs("checkpoints", exist_ok=True)

num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    train_acc = correct / total
    train_loss /= total
    
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    val_acc = correct / total
    val_loss /= total
    
    print(f"Epoch {epoch+1}/{num_epochs} "
      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
      
    torch.save(model.state_dict(), f"checkpoints/classifier_epoch_{epoch+1}.pth")
    
    
