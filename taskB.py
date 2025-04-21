import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# Normalization only; no data augmentation.
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1).cpu()
            image = transforms.functional.to_pil_image(image)
        if self.transform:
            image = self.transform(image)
        return image, label


class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)  # 64x64 -> 32x32
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)  # 32x32 -> 16x16
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Extra layer
            nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)  # 8x8 -> 4x4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_data(file_path):
    data = torch.load(file_path)
    return data['data'], data['labels']

class EarlyStopping:
    def __init__(self, patience=20, min_delta_factor=0.001):
        self.patience = patience
        self.min_delta = min_delta_factor
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            torch.save(model.state_dict(), path)
            self.counter = 0


def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, path, num_epochs, device, early_stopping):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    model.load_state_dict(torch.load(path))
    return model

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 150
    batch_size = 32
    learning_rate = 0.005

    subclass_files = [
        {"train": "TaskB/Model1/model1_train.pth", "test": "TaskB/Model1/model1_test.pth", "mapping": {34: 0, 137: 1, 159: 2, 173: 3, 201: 4}},
        {"train": "TaskB/Model2/model2_train.pth", "test": "TaskB/Model2/model2_test.pth", "mapping": {24: 0, 34: 1, 80: 2, 135: 3, 202: 4}},
        {"train": "TaskB/Model3/model3_train.pth", "test": "TaskB/Model3/model3_test.pth", "mapping": {124: 0, 125: 1, 130: 2, 173: 3, 202: 4}}
    ]

    for i, files in enumerate(subclass_files):
        print(f"\n======== Training Model {i + 1} ======")
        train_data, train_labels = load_data(files["train"])
        test_data, test_labels = load_data(files["test"])

        mapping = files["mapping"]
        train_labels_tensor = torch.tensor([mapping[int(label.item())] for label in train_labels], dtype=torch.long)
        test_labels_tensor = torch.tensor([mapping[int(label.item())] for label in test_labels], dtype=torch.long)

        train_dataset = CustomDataset(train_data, train_labels_tensor, transform=train_transform)
        test_dataset = CustomDataset(test_data, test_labels_tensor, transform=test_transform)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model selection
        if i == 2:  # Model 3
            model = CustomCNN(num_classes=5).to(device)
            weight_decay = 1e-3
        else:
            model = CustomCNN(num_classes=5).to(device)  # Assume this is the original model
            weight_decay = 5e-4

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        warmup_epochs = 5
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.05, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5)
        ], milestones=[warmup_epochs])

        early_stopping = EarlyStopping(patience=20, min_delta_factor=0.001)
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                            f'TaskB/Models/model{i + 1}.pth', num_epochs, device, early_stopping)

        print(f"--- Testing classifier for Domain {i + 1} ---")
        test_model(model, test_loader, device=device)


if __name__ == '__main__':
    main()
