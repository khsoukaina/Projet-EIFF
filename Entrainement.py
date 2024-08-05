import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError, ImageFile
from collections import Counter
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.enabled = True


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except UnidentifiedImageError:
            print(f"Unidentified image: {path}")
            sample = Image.new('RGB', (300, 300))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class MultiTaskEfficientNet(nn.Module):
    def __init__(self):
        super(MultiTaskEfficientNet, self).__init__()
        
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features

        
        self.model.classifier = nn.Identity()

        
        self.pylone_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.BatchNorm1d(2)
        )
        self.antenne_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.BatchNorm1d(2)
        )
        self.fh_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.BatchNorm1d(2)
        )

    def forward(self, x):
        features = self.model(x)
        pylone_out = self.pylone_fc(features)
        antenne_out = self.antenne_fc(features)
        fh_out = self.fh_fc(features)
        return pylone_out, antenne_out, fh_out


class EarlyStopping:
    def __init__(self, patience=5, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss


def train_model(model, criterion_pylone, criterion_antenne, criterion_fh, optimizer, train_loader, valid_loader, scheduler, num_epochs=100):
    train_losses = []
    valid_losses = []
    pylone_accuracies = []
    antenne_accuracies = []
    fh_accuracies = []
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_pylone, correct_antenne, correct_fh = 0, 0, 0
        total_pylone, total_antenne, total_fh = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            
            labels_pylone = (labels == 0).long().to(device)
            labels_antenne = (labels == 1).long().to(device)
            labels_fh = (labels == 2).long().to(device)

            pylone_outputs, antenne_outputs, fh_outputs = model(inputs)
            pylone_loss = criterion_pylone(pylone_outputs, labels_pylone)
            antenne_loss = criterion_antenne(antenne_outputs, labels_antenne)
            fh_loss = criterion_fh(fh_outputs, labels_fh)
            loss = pylone_loss + antenne_loss + fh_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            
            _, pred_pylone = torch.max(pylone_outputs, 1)
            _, pred_antenne = torch.max(antenne_outputs, 1)
            _, pred_fh = torch.max(fh_outputs, 1)
            correct_pylone += (pred_pylone == labels_pylone).sum().item()
            correct_antenne += (pred_antenne == labels_antenne).sum().item()
            correct_fh += (pred_fh == labels_fh).sum().item()
            total_pylone += labels_pylone.size(0)
            total_antenne += labels_antenne.size(0)
            total_fh += labels_fh.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        
        model.eval()
        val_loss = 0.0
        correct_pylone, correct_antenne, correct_fh = 0, 0, 0
        total_pylone, total_antenne, total_fh = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                labels_pylone = (labels == 0).long().to(device)
                labels_antenne = (labels == 1).long().to(device)
                labels_fh = (labels == 2).long().to(device)

                pylone_outputs, antenne_outputs, fh_outputs = model(inputs)
                pylone_loss = criterion_pylone(pylone_outputs, labels_pylone)
                antenne_loss = criterion_antenne(antenne_outputs, labels_antenne)
                fh_loss = criterion_fh(fh_outputs, labels_fh)
                loss = pylone_loss + antenne_loss + fh_loss
                val_loss += loss.item() * inputs.size(0)

                _, pred_pylone = torch.max(pylone_outputs, 1)
                _, pred_antenne = torch.max(antenne_outputs, 1)
                _, pred_fh = torch.max(fh_outputs, 1)
                correct_pylone += (pred_pylone == labels_pylone).sum().item()
                correct_antenne += (pred_antenne == labels_antenne).sum().item()
                correct_fh += (pred_fh == labels_fh).sum().item()
                total_pylone += labels_pylone.size(0)
                total_antenne += labels_antenne.size(0)
                total_fh += labels_fh.size(0)

        val_loss /= len(valid_loader.dataset)
        valid_losses.append(val_loss)
        pylone_accuracy = 100 * correct_pylone / total_pylone
        antenne_accuracy = 100 * correct_antenne / total_antenne
        fh_accuracy = 100 * correct_fh / total_fh
        pylone_accuracies.append(pylone_accuracy)
        antenne_accuracies.append(antenne_accuracy)
        fh_accuracies.append(fh_accuracy)

        
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr}')

        print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
              f'Pylone Accuracy: {pylone_accuracy:.2f}%, Antenne Accuracy: {antenne_accuracy:.2f}%, FH Accuracy: {fh_accuracy:.2f}%')

        scheduler.step(val_loss)  # Pass validation loss for ReduceLROnPlateau

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        end_time = time.time()
        print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")

    model.load_state_dict(torch.load('checkpoint.pth'))
    torch.save(model.state_dict(), 'model_trained.pth')
    print("Modèle sauvegardé sous 'model_trained.pth'")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(pylone_accuracies, label='Pylone Accuracy', color='blue')
    plt.plot(antenne_accuracies, label='Antenne Accuracy', color='green')
    plt.plot(fh_accuracies, label='FH Accuracy', color='red')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    train_dir = 'C:/Users/skhalil/Pictures/train'
    valid_dir = 'C:/Users/skhalil/Pictures/val'

    # Improved Data Augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.2),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageFolder(root=train_dir, transform=train_transforms)
    valid_dataset = CustomImageFolder(root=valid_dir, transform=valid_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskEfficientNet().to(device)

    class_counts = Counter([label for _, label in train_dataset.samples])
    total_train = sum(class_counts.values())

    class_weights_pylone = torch.tensor([total_train / class_counts[0], total_train / (total_train - class_counts[0])]).to(device)
    class_weights_antenne = torch.tensor([total_train / class_counts[1], total_train / (total_train - class_counts[1])]).to(device)
    class_weights_fh = torch.tensor([total_train / class_counts[2], total_train / (total_train - class_counts[2])]).to(device)

    criterion_pylone = nn.CrossEntropyLoss(weight=class_weights_pylone)
    criterion_antenne = nn.CrossEntropyLoss(weight=class_weights_antenne)
    criterion_fh = nn.CrossEntropyLoss(weight=class_weights_fh)

 
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    train_model(model, criterion_pylone, criterion_antenne, criterion_fh, optimizer, train_loader, valid_loader, scheduler)
