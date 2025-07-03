import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from KANUnet import KANUnet

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据生成器
class SyntheticMultiModalDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=256, path_features=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.path_features = path_features
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机路径特征 (模拟表格数据)
        path_data = np.random.rand(self.path_features).astype(np.float32)
        
        # 生成随机地图图像 (1通道)
        map_image = np.random.rand(1, self.image_size, self.image_size).astype(np.float32)
        
        # 生成随机目标值 (模拟回归任务)
        # 这里结合路径特征和图像特征生成目标值
        target = (path_data.sum() * 0.3 + map_image.mean() * 0.7).astype(np.float32)
        
        return (torch.from_numpy(path_data), 
                torch.from_numpy(map_image), 
                torch.from_numpy(np.array([target])))

# 2. 数据准备
def prepare_data(batch_size=4, path_features=10):
    # 创建完整数据集
    dataset = SyntheticMultiModalDataset(num_samples=200, 
                                       image_size=256,
                                       path_features=path_features)
    
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
    criterion = nn.MSELoss()  # 回归任务使用MSE损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 使用tqdm显示进度条
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for path_data, map_images, targets in loop:
            path_data = path_data.to(device)
            map_images = map_images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(path_data, map_images)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * path_data.size(0)
            
            # 更新进度条
            loop.set_postfix(loss=loss.item())
        
        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for path_data, map_images, targets in test_loader:
                path_data = path_data.to(device)
                map_images = map_images.to(device)
                targets = targets.to(device)
                outputs = model(path_data, map_images)
                test_loss += criterion(outputs, targets).item() * path_data.size(0)
        
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
        for path_data, map_images, targets in test_loader:
            path_data = path_data.to(device)
            map_images = map_images.to(device)
            targets = targets.to(device)
            
            outputs = model(path_data, map_images)
            
            # 保存结果用于可视化
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    # 可视化预测结果与真实值的对比
    visualize_results(predictions, targets_list)

def visualize_results(predictions, targets, num_samples=10):
    plt.figure(figsize=(15, 5))
    
    # 转换为numpy数组
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # 随机选择一些样本展示
    indices = np.random.choice(len(predictions), num_samples, replace=False)
    
    # 绘制预测值和真实值的对比
    plt.plot(targets[indices], 'bo', label='True values')
    plt.plot(predictions[indices], 'rx', label='Predictions')
    plt.legend()
    plt.title('Predictions vs True Values')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.show()
    
    # 绘制散点图展示相关性
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.show()

# 5. 主函数
def main():
    # 参数设置
    path_features = 10  # 路径特征的维度
    n_channels = 1     # 地图图像的通道数
    n_classes = 1      # 输出维度 (回归任务)
    input_size = path_features + n_classes  # KAN的输入尺寸
    
    # 准备数据
    train_loader, test_loader = prepare_data(batch_size=4, path_features=path_features)
    
    # 初始化模型
    model = KANUnet(input_size, n_channels, n_classes).to(device)
    
    # 打印模型结构
    print(model)
    
    # 训练模型
    trained_model = train_model(model, train_loader, test_loader, num_epochs=20)
    
    # 测试模型
    test_model(trained_model, test_loader)
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'kan_unet_model.pth')

if __name__ == '__main__':
    main()