import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

data_dir = "../../data/dataset_faces"
out_dir = "../../resultados/cnn_baseline"
os.makedirs(out_dir, exist_ok=True)

batch_size = 32
epochs = 10
learning_rate = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_acc_hist, val_acc_hist, train_loss_hist = [], [], []
best_val = 0.0
best_model_path = os.path.join(out_dir, "best_model.pt")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_acc = 100 * correct / total
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)
    train_loss_hist.append(avg_train_loss)

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f" Melhor CNN salva (val={best_val:.2f}%)")

    print(f"Epoch {epoch+1}/{epochs} | Loss {avg_train_loss:.4f} | Train {train_acc:.2f}% | Val {val_acc:.2f}%")

print("Treinamento finalizado ")
print(f"Melhor Val Acc: {best_val:.2f}%")
print("Modelo salvo em:", best_model_path)

ep = list(range(1, epochs + 1))

plt.figure()
plt.plot(ep, train_acc_hist, label="Train Acc")
plt.plot(ep, val_acc_hist, label="Val Acc")
plt.xlabel("Época")
plt.ylabel("Acurácia (%)")
plt.title("CNN Baseline - Acurácia por época")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_dir, "grafico_acuracia.png"), dpi=200)

plt.figure()
plt.plot(ep, train_loss_hist, label="Train Loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("CNN Baseline - Loss por época")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_dir, "grafico_loss.png"), dpi=200)

print("Gráficos salvos em:", out_dir)
