import torch
import os
import torch.nn as nn
from torchvision.models import ResNet18_Weights  # 导入权重枚举类
import torchvision.models as models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetClassifier, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def train_model(self, dataloader, criterion, optimizer, num_epochs=25):
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

    def predict(self, dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
        return predictions