import torch
import os
from torch.utils.data import Dataset, DataLoader
from read_file import CircuitParser
from transformer import TransformerEncoder, MLP
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(features, labels, train_ratio=0.8, batch_size=32):
    """
    划分数据集为训练集和测试集，并返回对应的 DataLoader。

    参数：
        features (torch.Tensor): 特征张量，形状为 (N, ...)。
        labels (torch.Tensor): 标签张量，形状为 (N, ...)。
        train_ratio (float): 训练集占比，默认 0.8。
        batch_size (int): 每个批次的大小，默认 32。

    返回：
        tuple: train_loader 和 test_loader。
    """
    # 使用 sklearn 的 train_test_split 划分数据集
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, train_size=train_ratio, random_state=42
    )

    # 将划分后的数据转换为 Dataset 对象
    train_dataset = MyDataset(features_train, labels_train)
    test_dataset = MyDataset(features_test, labels_test)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train(model, train_loader, test_loader, optimizer, criterion, epochs, device):
    model.train()

    writer = SummaryWriter()  # 初始化 TensorBoard writer

    for epoch in range(epochs):
        running_loss = 0.0
        total_relative_error = 0.0

        # Training loop
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute relative error for training
            relative_error = torch.abs((outputs - labels) / labels).mean().item()
            total_relative_error += relative_error

        avg_train_loss = running_loss / len(train_loader)
        avg_train_mre = total_relative_error / len(train_loader)

        # Validation loop
        model.eval()
        test_loss = 0.0
        test_relative_error = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Compute relative error for testing
                relative_error = torch.abs((outputs - labels) / labels).mean().item()
                test_relative_error += relative_error

        avg_test_loss = test_loss / len(test_loader)
        avg_test_mre = test_relative_error / len(test_loader)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Test', avg_test_loss, epoch)
        writer.add_scalar('MRE/Train', avg_train_mre, epoch)
        writer.add_scalar('MRE/Test', avg_test_mre, epoch)

        # Print metrics for the current epoch
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train MRE: {avg_train_mre:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}, Test MRE: {avg_test_mre:.4f}")

    writer.close()

def predict(model, dataloader, device):
    """
    使用模型对数据进行预测,同时返回预测结果与真实结果的平均相对误差(MRE)。

    参数：
        model (torch.nn.Module): 要评估的模型。
        dataloader (DataLoader): 数据加载器，包含输入和标签。
        device (torch.device): 计算设备(CPU 或 GPU)。

    返回：
        tuple: 包含预测结果 (Tensor) 和平均相对误差 (float)。
    """
    model.eval()
    predictions = []
    total_relative_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())

            # 计算相对误差
            relative_error = torch.abs((outputs - labels) / labels).sum().item()
            total_relative_error += relative_error
            total_samples += labels.size(0)

    predictions = torch.cat(predictions)
    mean_relative_error = total_relative_error / total_samples

    print(f"Mean Relative Error (MRE): {mean_relative_error:.4f}")

    return predictions, mean_relative_error



if __name__ == "__main__":
    # Directory containing the YAML files
    directory = "/home/liuhe/large_model/wire_paths"

    all_tensors = []
    all_labels = []
    max_length = 0
    count = 0
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".yml"):
            file_path = os.path.join(directory, filename)
            parser = CircuitParser(file_path)
            parser.load_data()
            parser.parse_data()
            combined_tensor = parser.get_combined_tensor()
            incr_tensor, incr_sum = parser.get_incr_tensor()

            hashcode = parser.generate_hash()
            print(parser.point_list)
            # Update the maximum length
            max_length = max(max_length, combined_tensor.size(1))

            # Collect the tensors and labels
            all_tensors.append(combined_tensor)
            all_labels.append(incr_sum)

    max_length = 256
    # Pad all tensors to the maximum length
    padded_tensors = CircuitParser.pad_tensors(all_tensors, max_length)

    # Stack tensors into a 3D tensor
    features_tensor = torch.stack(padded_tensors).permute(0, 2, 1)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)

    print(f"Final Tensor Shape: {features_tensor.shape}")
    print(f"Labels Tensor Shape: {labels_tensor.shape}")
    print("Final Tensor:")
    print(features_tensor)
    print("Labels Tensor:")
    print(labels_tensor)
    print("Debug")
    
    # 超参数
    input_dim = 3  # 每个节点的特征数
    hidden_dim = 3
    num_layers = 2
    num_heads = 3
    mlp_hidden_dim = 16
    output_dim = 1
    epochs = 100
    learning_rate = 0.0005

    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    train_loader, test_loader = load_data(features_tensor, labels_tensor)

    # 定义模型
    transformer_encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads)
    mlp = MLP(hidden_dim, mlp_hidden_dim, output_dim)
    model = nn.Sequential(transformer_encoder, mlp).to(device)

    # 定义优化器和损失函数
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)  # T_max 是一个周期的迭代次数
    criterion = nn.MSELoss().to(device)

    # 训练模型
    train(model, train_loader, test_loader, optimizer, criterion, epochs, device)

    # 预测
    predictions, mre = predict(model, test_loader, device)
    print(predictions)