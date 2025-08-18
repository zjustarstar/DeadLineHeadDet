import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from classifier_network import ResNetClassifier

class DeadlineDataset(Dataset):
    """
    线头区域数据集
    """
    def __init__(self, data_dir, class_name, transform=None):
        """
        初始化数据集

        Args:
            data_dir: 数据目录
            class_name: 类别名称（positive或negative）
            transform: 数据增强转换
        """
        self.data_dir = os.path.join(data_dir, class_name)
        self.class_name = class_name
        self.transform = transform
        # 类别
        if class_name == "n":
            self.label = 0
        elif class_name == "ss":
            self.label = 1
        elif class_name == "wv":
            self.label = 2
        elif class_name == "t":
            self.label = 3
        else:
            self.label = 4
        
        # 获取所有图像文件
        self.image_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 读取图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, self.label

def create_data_transforms():
    """
    创建数据增强转换
    """
    # 获取区域大小
    input_size = 100
    
    # 训练集转换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30, fill=(255, 255, 255))
        ], p=0.6),  # 整体有50%概率执行其中任一变换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集转换（不包含数据增强）
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_data(data_dir):
    """
    准备训练和验证数据

    Returns:
        训练数据加载器和验证数据加载器
    """
    batch_size = 25
    
    # 创建数据增强转换
    train_transform, val_transform = create_data_transforms()
    
    # 创建正样本和负样本数据集
    dataset_n = DeadlineDataset(data_dir, "n", transform=train_transform)
    dataset_ss = DeadlineDataset(data_dir, "ss", transform=train_transform)
    dataset_wv = DeadlineDataset(data_dir, "wv", transform=train_transform)
    dataset_t = DeadlineDataset(data_dir, "t", transform=train_transform)
    dataset_other = DeadlineDataset(data_dir, "other", transform=train_transform)
    
    # 合并数据集
    dataset = torch.utils.data.ConcatDataset([dataset_n, dataset_ss, dataset_wv, dataset_t, dataset_other])
    print(f"加载了 {len(dataset)} 样本")
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 为验证集设置不同的转换
    val_dataset.dataset.transform = val_transform
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device):
    """
    训练模型

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        args: 命令行参数

    Returns:
        训练历史记录
    """
    # 获取配置
    epochs = 100
    learning_rate = 0.01
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 训练循环
    min_loss = 1000000
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm显示进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})
        
        # 计算训练集平均损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        if train_loss<min_loss:
            print(f"发现更优模型！Loss: {train_loss:.4f} (之前最佳: {min_loss:.4f})")
            min_loss = train_loss
            
            # 在保存模型前检查最后一层权重
            fc_weight = model.resnet.fc.weight.data
            print(f"\n保存模型前最后一层权重统计:")
            print(f"形状: {fc_weight.shape}")
            print(f"均值: {fc_weight.mean().item():.6f}")
            print(f"标准差: {fc_weight.std().item():.6f}")
            print(f"最大值: {fc_weight.max().item():.6f}")
            print(f"最小值: {fc_weight.min().item():.6f}")
            
            # 检查是否所有权重都相同
            if torch.allclose(fc_weight[0], fc_weight[1], atol=1e-5) and torch.allclose(fc_weight[0], fc_weight[2], atol=1e-5):
                print("警告: 最后一层的权重几乎完全相同，这可能导致所有类别的预测结果一致!")
                
            # 保存模型状态字典
            state_dict = model.state_dict()
            print(f"\n状态字典包含以下键:")
            for key in state_dict.keys():
                print(f"  - {key}")
                
            torch.save(state_dict, 'models\\classifier.pth')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 不计算梯度
        with torch.no_grad():
            # 使用tqdm显示进度条
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct / val_total})
        
        # 计算验证集平均损失和准确率
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

def plot_history(history, save_path=None):
    """
    绘制训练历史记录

    Args:
        history: 训练历史记录
        save_path: 保存路径，如果为None则显示图像
    """
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path)
        print(f"训练历史记录已保存至: {save_path}")
    else:
        plt.show()

def prepare_training_data(data_dir):
    """
    准备训练数据，将rect_region目录中的图像分类到正样本和负样本目录

    Args:
        args: 命令行参数
    """
    # 检查rect_region目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 区域目录不存在: {data_dir}")
        return
    
    # 获取所有图像文件
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"警告: 区域目录中没有找到图像文件: {data_dir}")
        return
    
    print(f"找到 {len(image_files)} 个样本文件")


def main():
    data_dir = ".\\sample"
    # prepare_training_data(data_dir)
    
    # 训练模型
    # 准备数据
    train_loader, val_loader = prepare_data(data_dir)

    # 创建模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = ".\\models\\classifier.pth"
    model = ResNetClassifier(num_classes=5)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")

    # 训练模型
    history = train_model(model, train_loader, val_loader, device=device)

    # 绘制训练历史记录
    history_path = "train_history.png"
    plot_history(history, save_path=history_path)

if __name__ == "__main__":
    main()