import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# 直接引用第一段代码中的类和函数
from taskB import CustomCNN, CustomDataset, test_transform, load_data


# 评估每个类别的准确率
def evaluate_per_class(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算总体准确率
    total_correct = sum(1 for p, t in zip(all_preds, all_labels) if p == t)
    total_samples = len(all_labels)
    overall_accuracy = 100 * total_correct / total_samples
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")

    # 计算每个类别的准确率
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = []
    for i in range(len(class_names)):
        true_positives = cm[i, i]
        total_samples_class = cm[i].sum()
        accuracy = 100 * true_positives / total_samples_class if total_samples_class > 0 else 0
        per_class_accuracy.append(accuracy)
        print(f"Accuracy for {class_names[i]} (Label {i}): {accuracy:.2f}% ({true_positives}/{total_samples_class})")

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Model 3")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="black" if cm[i, j] < cm.max() / 2 else "white")
    plt.show()

    return per_class_accuracy, cm


# 主验证函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    # Model 3的类别名称和映射
    class_names = ["Collie", "Golden Retriever", "Boxer", "Chihuahua", "African Hunting Dog"]
    mapping = {124: 0, 125: 1, 130: 2, 173: 3, 202: 4}

    # 加载测试数据
    test_data, test_labels = load_data("../TaskB/Model3/model3_test.pth")
    test_labels_mapped = torch.tensor([mapping[int(label.item())] for label in test_labels], dtype=torch.long)
    test_dataset = CustomDataset(test_data, test_labels_mapped, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载 Model 3
    model = CustomCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load("../TaskB/Models/model3.pth", map_location=device))
    print("Model 3 loaded successfully.")

    # 评估每个类别的准确率
    print("\nEvaluating Model 3 on Test Set:")
    per_class_accuracy, cm = evaluate_per_class(model, test_loader, device, class_names)


if __name__ == "__main__":
    main()
