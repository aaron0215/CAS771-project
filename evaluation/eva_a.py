import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from torch.utils.data import DataLoader
import numpy as np

from main import CustomDataset, CustomCNN, test_transform, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subclass_files = [
    {
        "test": "../Model1/model1_test.pth",
        "model_path": "../Models/model1.pth",
        "mapping": {0: 0, 40: 1, 10: 2, 20: 3, 30: 4}
    },
    {
        "test": "../Model2/model2_test.pth",
        "model_path": "../Models/model2.pth",
        "mapping": {1: 0, 41: 1, 11: 2, 21: 3, 31: 4}
    },
    {
        "test": "../Model3/model3_test.pth",
        "model_path": "../Models/model3.pth",
        "mapping": {32: 0, 2: 1, 42: 2, 12: 3, 22: 4}
    }
]


def evaluate_model(model, test_loader, num_classes):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return all_labels, all_probs, accuracy, precision, recall, f1


def plot_roc_curve(y_true, y_score, num_classes, model_index):
    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {model_index + 1} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def main():
    for i, files in enumerate(subclass_files):
        print(f"\n===== Evaluating Model {i + 1} =====")

        test_data, test_labels = load_data(files["test"])
        mapping = files["mapping"]
        test_labels_tensor = torch.tensor([mapping[label] for label in test_labels], dtype=torch.long)

        test_dataset = CustomDataset(test_data, test_labels_tensor, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = CustomCNN(num_classes=5).to(device)
        model.load_state_dict(torch.load(files["model_path"], map_location=device))

        y_true, y_score, acc, prec, rec, f1 = evaluate_model(model, test_loader, num_classes=5)

        plot_roc_curve(y_true, y_score, num_classes=5, model_index=i)


if __name__ == "__main__":
    main()
