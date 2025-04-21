from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    AutoAugment(AutoAugmentPolicy.CIFAR10),
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
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if isinstance(image, torch.Tensor):
            image = transforms.functional.to_pil_image(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomCNN(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.2):
        super(CustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )


        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    labels = raw_data['labels']
    return data, labels


class EarlyStopping:
    def __init__(self, patience=10, min_delta_factor=0.005):
        self.patience = patience
        self.min_delta_factor = min_delta_factor
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_wts = None

    def check_stop(self, model, loss):
        min_delta = self.min_delta_factor * self.best_loss if self.best_loss < float('inf') else 0.01
        if loss < self.best_loss - min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_model_wts = model.state_dict()
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_model(model, train_loader, criterion, optimizer, scheduler, save_path, num_epochs, device):
    model.to(device)
    early_stopper = EarlyStopping()
    loss_history = []

    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f}")

        scheduler.step()
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()

        if early_stopper.check_stop(model, epoch_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved at {save_path}")
    return model


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    num_epochs = 100
    batch_size = 32
    learning_rate = 0.01
    weight_decay = 1e-4  # L2 re


    subclass_files = [
        {
            "train": "Model1/model1_train.pth",
            "test": "Model1/model1_test.pth",
            "mapping": {0: 0, 40: 1, 10: 2, 20: 3, 30: 4}
        },
        {
            "train": "Model2/model2_train.pth",
            "test": "Model2/model2_test.pth",
            "mapping": {1: 0, 41: 1, 11: 2, 21: 3, 31: 4}
        },
        {
            "train": "Model3/model3_train.pth",
            "test": "Model3/model3_test.pth",
            "mapping": {32: 0, 2: 1, 42: 2, 12: 3, 22: 4}
        }
    ]


    for i, files in enumerate(subclass_files):
        print(f"\n======== Training Model {i + 1} ======")
        # Load data.
        train_data, train_labels = load_data(files["train"])
        test_data, test_labels = load_data(files["test"])


        # Map labels using the provided mapping.
        mapping = files["mapping"]
        train_labels_tensor = torch.tensor([mapping[label] for label in train_labels], dtype=torch.long)
        test_labels_tensor = torch.tensor([mapping[label] for label in test_labels], dtype=torch.long)

        train_dataset = CustomDataset(train_data, train_labels_tensor, transform=train_transform)
        test_dataset = CustomDataset(test_data, test_labels_tensor, transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = CustomCNN(num_classes=5).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        warmup_epochs = 5
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5)
        ], milestones=[warmup_epochs])

        # Train the model.
        model = train_model(model, train_loader, criterion, optimizer, scheduler,
                            f'Models/model{i + 1}.pth', num_epochs, device)

        # Test the model.
        print(f"--- Testing classifier for Domain {i + 1} ---")
        test_model(model, test_loader, device=device)


if __name__ == '__main__':
    main()