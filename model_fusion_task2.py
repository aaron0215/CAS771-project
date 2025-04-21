import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

# --- Custom Dataset ---
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

# --- Model Definition ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
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

# --- Data Loading Helpers ---
def load_data(file_path):
    data = torch.load(file_path)
    return data['data'], data['labels']

def load_combined_data(subclass_files, train=True):
    all_data = []
    all_labels = []
    for i, files in enumerate(subclass_files):
        file_path = files["train"] if train else files["test"]
        data, labels = load_data(file_path)
        mapping = files["mapping"]
        offset = i * 5
        # Convert each label to an int before mapping
        new_labels = [mapping[int(label)] + offset for label in labels]
        all_data.append(data)
        all_labels.extend(new_labels)
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    return all_data, all_labels

# --- Early Stopping ---
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

# --- Accuracy and Training/Testing Functions ---
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
    return accuracy

# --- Fuse Models Function ---
def fuse_models(model_paths, device):
    models = []
    for path in model_paths:
        # Each individual model was trained for 5 classes.
        model = CustomCNN(num_classes=5)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    # Create a merged model that can classify 15 classes (3 domains x 5 classes).
    merged_model = CustomCNN(num_classes=15).to(device)
    merged_state = merged_model.state_dict()

    # Average parameters for all layers except the final classifier linear layer.
    for key in merged_state.keys():
        if key.startswith("classifier.1."):
            continue
        params = [m.state_dict()[key] for m in models]
        merged_state[key] = sum(params) / len(params)

    # Concatenate the classifier weights and biases.
    final_weight = merged_state["classifier.1.weight"]
    final_bias = merged_state["classifier.1.bias"]
    start = 0
    for m in models:
        m_state = m.state_dict()
        w = m_state["classifier.1.weight"]
        b = m_state["classifier.1.bias"]
        final_weight[start:start+5, :] = w
        final_bias[start:start+5] = b
        start += 5
    merged_state["classifier.1.weight"] = final_weight
    merged_state["classifier.1.bias"] = final_bias

    merged_model.load_state_dict(merged_state)
    print("Models fused successfully.")
    return merged_model

# --- Main Function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 150
    batch_size = 32
    learning_rate = 0.003
    warmup_epochs = 10

    os.makedirs("TaskB/Models", exist_ok=True)

    subclass_files = [
        {"train": "TaskB/Model1/model1_train.pth", "test": "TaskB/Model1/model1_test.pth",
         "mapping": {34: 0, 137: 1, 159: 2, 173: 3, 201: 4}},
        {"train": "TaskB/Model2/model2_train.pth", "test": "TaskB/Model2/model2_test.pth",
         "mapping": {24: 0, 34: 1, 80: 2, 135: 3, 202: 4}},
        {"train": "TaskB/Model3/model3_train.pth", "test": "TaskB/Model3/model3_test.pth",
         "mapping": {124: 0, 125: 1, 130: 2, 173: 3, 202: 4}}
    ]

    # --- Fuse the Individual Models ---
    model_paths = [f"TaskB/Models/model{i + 1}.pth" for i in range(3)]
    merged_model = fuse_models(model_paths, device)

    # --- Fine-tune the Fused Model on the Combined Dataset ---
    print("\n======== Fine-tuning Fused Model on Combined Dataset ========")
    combined_train_data, combined_train_labels = load_combined_data(subclass_files, train=True)
    combined_test_data, combined_test_labels = load_combined_data(subclass_files, train=False)

    combined_train_dataset = CustomDataset(combined_train_data, combined_train_labels, transform=train_transform)
    combined_test_dataset = CustomDataset(combined_test_data, combined_test_labels, transform=test_transform)

    train_size = int(0.8 * len(combined_train_dataset))
    val_size = len(combined_train_dataset) - train_size
    train_set, val_set = random_split(combined_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.SGD(merged_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.05, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5)
    ], milestones=[warmup_epochs])
    early_stopping = EarlyStopping(patience=30, min_delta_factor=0.001)
    merged_model_save_path = "TaskB/Models/merged_model_finetuned.pth"
    merged_model = train_model(merged_model, train_loader, val_loader, criterion, optimizer, scheduler,
                               merged_model_save_path, num_epochs, device, early_stopping)

    print("\n--- Testing Fused Model on Combined Test Set ---")
    test_model(merged_model, test_loader, device)

    print(f"\nFine-tuned merged model saved at {merged_model_save_path}")

if __name__ == '__main__':
    main()
