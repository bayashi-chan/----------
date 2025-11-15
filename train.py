import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# デバイス（MPS = Mac のGPU）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ========= Dataset ==========
class DigitDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("L")  # 白黒画像に変換

        if self.transform:
            image = self.transform(image)

        return image, label

# ========= CNNモデル ===========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ========= 学習処理 =============
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

train_dataset = DigitDataset(
    "train_master.csv",
    "train",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):  # とりあえず3エポック
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ========= 推論 & 提出ファイル作成 ===========
import re

def extract_number(filename):
    return int(re.findall(r'\d+', filename)[0])

test_files = sorted(os.listdir("test"), key=extract_number)

preds = []
model.eval()
with torch.no_grad():
    for img_name in test_files:
        img_path = os.path.join("test", img_name)
        image = Image.open(img_path).convert("L")
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        pred = output.argmax(dim=1).item()
        preds.append([img_name, pred])

df_sub = pd.DataFrame(preds, columns=["file_name", "label"])
df_sub.to_csv("submission.csv", index=False)

print("submission.csv を作成しました！")