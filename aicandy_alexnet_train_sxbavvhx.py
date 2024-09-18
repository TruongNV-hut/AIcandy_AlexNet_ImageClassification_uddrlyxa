"""

@author:  AIcandy 
@website: aicandy.vn

"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from aicandy_model_src_rimrtnqo.aicandy_alexnet_model_dndvgvhk import AlexNet



# Command:
# python aicandy_alexnet_train_sxbavvhx.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_uoebddte/aicandy_model_pth_luveqrpt.pth


def train(train_dir, num_epochs, batch_size, model_path, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training with: ", device)
    
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(20),      # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Data augmentation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Lưu nhãn và id lớp
    with open('label.txt', 'w') as f:
        for idx, class_name in enumerate(dataset.classes):
            f.write(f'{idx}: {class_name}\n')

    num_classes = len(dataset.classes)
    model = AlexNet(num_classes=num_classes).to(device)
    
    # Thử SGD với momentum
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Huấn luyện mô hình
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Đánh giá trên validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model with accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to save the best model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')


    args = parser.parse_args()
    train(args.train_dir, args.num_epochs, args.batch_size, args.model_path, args.learning_rate)
