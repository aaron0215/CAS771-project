import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    AutoAugment(AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
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


def load_data(data_path):
    raw = torch.load(data_path)
    data = raw['data']
    labels = raw['labels']
    return data, labels



def load_combined_data(subclass_files, train=True):
    all_data = []
    all_labels = []
    for i, files in enumerate(subclass_files):
        file_path = files["train"] if train else files["test"]
        data, labels = load_data(file_path)
        mapping = files["mapping"]
        offset = i * 5
        new_labels = [mapping[label] + offset for label in labels]
        all_data.append(data)
        all_labels.extend(new_labels)

    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    return all_data, all_labels


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
            self.best_model_wts = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.to(device)
    early_stopper = EarlyStopping(patience=10)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
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
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f}")
        scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        if early_stopper.check_stop(model, epoch_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    model.load_state_dict(best_model_wts)
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
    acc = correct / total
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return acc


def fuse_models(model_paths, device):
    models = []
    for path in model_paths:
        model = CustomCNN(num_classes=5)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    merged_model = CustomCNN(num_classes=15).to(device)
    merged_state = merged_model.state_dict()

    for key in merged_state.keys():
        if key.startswith('classifier.3'):
            continue
        params = [m.state_dict()[key] for m in models]
        merged_state[key] = sum(params) / len(params)

    final_weight = merged_state['classifier.3.weight']
    final_bias = merged_state['classifier.3.bias']
    start = 0
    for m in models:
        m_state = m.state_dict()
        w = m_state['classifier.3.weight']
        b = m_state['classifier.3.bias']
        final_weight[start:start + 5, :] = w
        final_bias[start:start + 5] = b
        start += 5
    merged_state['classifier.3.weight'] = final_weight
    merged_state['classifier.3.bias'] = final_bias

    merged_model.load_state_dict(merged_state)
    print("Models fused successfully.")
    return merged_model


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = ["Models/model1.pth", "Models/model2.pth", "Models/model3.pth"]
    merged_model = fuse_models(model_paths, device)

    combined_train_data, combined_train_labels = load_combined_data(subclass_files, train=True)
    combined_test_data, combined_test_labels = load_combined_data(subclass_files, train=False)

    train_dataset = CustomDataset(combined_train_data, combined_train_labels, transform=train_transform)
    test_dataset = CustomDataset(combined_test_data, combined_test_labels, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(merged_model.parameters(), lr=0.01, weight_decay=1e-4)
    warmup_epochs = 5
    total_epochs = 150
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-5)
    ], milestones=[warmup_epochs])

    print("Fine-tuning fused model on combined dataset...")
    merged_model = train_model(merged_model, train_loader, criterion, optimizer, scheduler, total_epochs, device)

    print("Evaluating fused model on test set...")
    test_model(merged_model, test_loader, device)

    os.makedirs("Models", exist_ok=True)
    torch.save(merged_model.state_dict(), "Models/merged_model_finetuned.pth")
    print("Fine-tuned merged model saved at Models/merged_model_finetuned.pth")


if __name__ == '__main__':
    main()
