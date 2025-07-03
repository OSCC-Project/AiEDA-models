import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from iAFF_UNet import UNet_pytorch_aff

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据生成器
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机输入图像 (6通道)
        input_img = np.random.rand(6, self.image_size, self.image_size).astype(np.float32)
        
        # 生成随机目标图像 (1通道)
        # 这里使用简单的处理模拟真实情况 - 实际应用中应替换为真实数据
        target_img = np.expand_dims(input_img[0] * 0.3 + input_img[1] * 0.5 - input_img[2] * 0.2, axis=0)
        target_img = np.clip(target_img, 0, 1)  # 限制在0-1范围
        
        return torch.from_numpy(input_img), torch.from_numpy(target_img)

# 2. 数据准备
def prepare_data(batch_size=4):
    # 创建完整数据集
    dataset = SyntheticDataset(num_samples=200, image_size=256)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 3. 训练函数
def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 使用tqdm显示进度条
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, targets in loop:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # 更新进度条
            loop.set_postfix(loss=loss.item())
        
        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item() * inputs.size(0)
        
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.show()
    
    return model

# 4. 测试和预测函数
def test_model(model, test_loader):
    model.eval()
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # 保存结果用于可视化
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    # 可视化一些结果
    visualize_results(predictions, targets_list)

def visualize_results(predictions, targets, num_samples=3):
    plt.figure(figsize=(15, 5))
    
    for i in range(num_samples):
        # 显示目标图像
        plt.subplot(2, num_samples, i+1)
        plt.imshow(targets[i][0], cmap='gray')
        plt.title(f'Target {i+1}')
        plt.axis('off')
        
        # 显示预测图像
        plt.subplot(2, num_samples, num_samples+i+1)
        plt.imshow(predictions[i][0], cmap='gray')
        plt.title(f'Prediction {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 5. 主函数
def main():
    # 准备数据
    train_loader, test_loader = prepare_data(batch_size=4)
    
    # 初始化模型
    model = UNet_pytorch_aff().to(device)
    
    # 打印模型结构
    print(model)
    
    # 训练模型
    trained_model = train_model(model, train_loader, test_loader, num_epochs=20)
    
    # 测试模型
    test_model(trained_model, test_loader)
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'resunet_inception_model.pth')

if __name__ == '__main__':
    main()

    