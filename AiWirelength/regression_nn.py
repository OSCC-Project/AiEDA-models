from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import time  # 导入时间模块
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.optim as optim

import warnings
from sklearn.exceptions import ConvergenceWarning

# 忽略 ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def load_and_prepare_data(filepath):
    """
    加载数据并准备特征和目标变量，去除 pin_num = 3 的样本
    """
    # 读取数据
    data = pd.read_csv(filepath)
    
    # 去除 pin_num = 3 的样本
    data = data[data['pin_num'] != 3]
    
    # 计算目标变量
    data['target'] = data['rsmt'] / data['hpwl']
    
    # 选择特征
    features = [
                'pin_num', 
                'lness', 
                'x_entropy',
                'y_entropy',
                'x_ratio_nn_dist', 
                'y_ratio_nn_dist',
                'aspect_ratio', 
                'bbox_area', 
                'hpwl',
                'x_std_nn_dist',
                'y_std_nn_dist',
                ]
    X = data[features]
    y = data['target']
    
    return X, y, data, features

def create_engineered_features(data):
    """
    基于物理意义创建新特征
    """   
    
    # 1. 布线资源
    data['route_ratio_x'] = data['bbox_width'] / (data['bbox_area'])
    data['route_ratio_y'] = data['bbox_height']/ (data['bbox_area'])

    # 2. 标准差比例
    data['std_nn_ratio'] = data['x_std_nn_dist'] / data['y_std_nn_dist']
    
    return data

def weighted_resample(X_train, y_train, n_samples=None, random_state=42, power=3):
    """
    基于目标变量值的非线性放大权重进行加权采样

    参数:
        X_train: pd.DataFrame 或 np.ndarray
            训练集的特征
        y_train: pd.Series 或 np.ndarray
            训练集的目标变量
        n_samples: int, 默认值为 None
            重采样后的样本数量。如果为 None, 则保持原始样本数量。
        random_state: int, 默认值为 42
            随机种子，确保结果可复现
        power: int, 默认值为 2
            权重放大的幂次，值越大，目标值较大的样本被采样的概率越高。

    返回:
        X_resampled: pd.DataFrame 或 np.ndarray
            重采样后的训练集特征
        y_resampled: pd.Series 或 np.ndarray
            重采样后的训练集目标变量
    """
    # 将特征和目标变量合并为一个 DataFrame
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.columns = list(X_train.columns) + ['target']  # 添加目标变量列名

    # 如果未指定采样数量，则保持原始样本数量
    if n_samples is None:
        n_samples = len(train_data)

    # 计算采样权重：目标值非线性放大
    weights = (train_data['target'] ** power) / (train_data['target'] ** power).sum()

    # 使用加权采样
    resampled_data = train_data.sample(n=n_samples, replace=True, weights=weights, random_state=random_state)

    # 分离特征和目标变量
    X_resampled = resampled_data.drop(columns=['target'])
    y_resampled = resampled_data['target']

    return X_resampled, y_resampled

# 绘制 MAE 曲线
def plot_mae_curve(mae_values, save_path):


    # 从第 1000 次迭代开始截取损失
    start_epoch = 1000
    mae_values = mae_values[start_epoch:]

    # 创建对应的横轴范围
    epochs = range(start_epoch, start_epoch + len(mae_values))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mae_values, label='Validation MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Validation MAE Curve')
    plt.legend()
    plt.grid()
    print(f"MAE curve saved to {save_path}")
    plt.savefig(save_path)
    plt.close()

# 绘制残差图
def plot_residuals(y_true, y_pred, save_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid()
    plt.savefig(save_path)
    print(f"Residual plot saved to {save_path}")
    plt.close()

# 保存验证集真实值和预测值到 CSV 文件
def save_predictions_to_csv(y_true, y_pred, save_path):
    df = pd.DataFrame({'True Values': y_true, 'Predicted Values': y_pred})
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

def plot_resampled_target_distribution(y_train, y_train_resampled):
    """
    绘制采样前后的目标变量分布
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制采样前的目标变量分布
    plt.hist(y_train, bins=50, alpha=0.5, label='Before Resampling')
    
    # 绘制采样后的目标变量分布
    plt.hist(y_train_resampled, bins=50, alpha=0.5, label='After Resampling')
    
    # 添加标题和坐标轴标签
    plt.title("Target Variable Distribution Before and After Resampling")
    plt.xlabel("Target Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig('fig/regression_target_distribution_resampled.png')
    print(f"重采样前的训练集样本数量: {len(y_train)}")
    print(f"重采样后的训练集样本数量: {len(y_train_resampled)}")


# 数据预处理函数
def prepare_data(filepath, engineered_features):
    """
    加载数据、创建新特征、划分数据集并进行归一化处理。
    """
    # 加载和准备数据
    X, y, data, features = load_and_prepare_data(filepath)

    # 创建新的特征
    data = create_engineered_features(data)

    # 更新特征列表
    features.extend(engineered_features)
    X = data[features]  # 更新 X

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 对训练集进行重复采样（平衡目标变量分布）
    X_train_resampled, y_train_resampled = weighted_resample(X_train, y_train, n_samples=40000, power=2)
    plot_resampled_target_distribution(y_train, y_train_resampled)

    # 对特征进行归一化（避免数据泄露）
    scaler = MinMaxScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # 返回训练集和测试集
    return X_train_resampled, X_test, y_train_resampled, y_test, scaler

# 数据转换为 PyTorch 张量
def convert_to_tensor(X_train, X_test, y_train, y_test, device):
    """
    将数据转换为 PyTorch 张量并移动到指定设备。
    """
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).to(device)
    return X_train, X_test, y_train, y_test

# 测试模型性能
def evaluate_model(model, X_test, y_test, device):
    """
    测试模型性能并返回性能指标。
    """
    model.eval()
    with torch.no_grad():
        # 记录预测时间
        start_time = time.time()
        y_pred = model(X_test).squeeze()
        end_time = time.time()
        prediction_time = end_time - start_time

        # 转换为 NumPy 数组
        y_test_np = y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # 计算性能指标
        mse = mean_squared_error(y_test_np, y_pred_np)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        r2 = r2_score(y_test_np, y_pred_np)
        rmse = np.sqrt(mse)

    y_test_true = y_test.cpu().numpy().flatten()
    y_test_pred = y_pred.cpu().numpy().flatten()
    # 保存预测结果
    save_predictions_to_csv(y_test_true, y_test_pred, 'csv/test_predictions.csv')
    # 绘制残差图
    plot_residuals(y_test_true, y_test_pred, 'fig/mlp_residual_plot.png')

    return {
        "预测时间 (秒)": prediction_time,
        "均方误差 (MSE)": mse,
        "平均绝对误差 (MAE)": mae,
        "决定系数 (R²)": r2,
        "均方根误差 (RMSE)": rmse
    }


# 训练 MLP 模型
def train_mlp(model, criterion, optimizer, X_train, y_train, num_epochs, device):
    """
    训练 MLP 模型并返回训练损失。
    """
    train_loss = []
    mae_values = []  # 用于记录每个 epoch 的 MAE


    for epoch in range(num_epochs):
        # 前向传播
        model.train()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        # 计算 MAE
        with torch.no_grad():
            y_train_np = y_train.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            mae = mean_absolute_error(y_train_np, outputs_np)
            mae_values.append(mae)


        # 打印训练进度
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train MAE: {mae:.4f}")

    return train_loss, mae_values

# 绘制损失曲线
def plot_loss_curve(train_loss, val_loss, output_path):
    """
    绘制训练和验证损失曲线并保存为图片。
    """
    # 从第 1500 次迭代开始截取损失
    start_epoch = 1500
    train_loss = train_loss[start_epoch:]
    val_loss = val_loss[start_epoch:]

    # 创建对应的横轴范围
    epochs = range(start_epoch, start_epoch + len(train_loss))

    # 绘制曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Curve")
    plt.savefig(output_path)

# 主函数
def main(filepath):
    """
    主函数：整合所有步骤
    """
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据准备
    engineered_features = ['std_nn_ratio', 'route_ratio_x', 'route_ratio_y']
    X_train, X_test, y_train, y_test, scaler = prepare_data(filepath, engineered_features)

    # 转换为 PyTorch 张量
    X_train, X_test, y_train, y_test = convert_to_tensor(X_train, X_test, y_train, y_test, device)

    # 超参数
    input_size = X_train.shape[1]
    hidden_sizes = [256, 128, 64, 32, 16]
    output_size = 1
    learning_rate = 0.0001
    num_epochs = 7000

    # 初始化模型
    model = MLP(input_size, hidden_sizes, output_size).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_loss, mae_values = train_mlp(model, criterion, optimizer, X_train, y_train, num_epochs, device)

    # 绘制 MAE 曲线
    plot_mae_curve(mae_values, 'fig/train_mae_curve.png')

    # 测试模型性能
    test_metrics = evaluate_model(model, X_test, y_test, device)
    print("测试集性能指标：")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


# 调用主函数
if __name__ == "__main__":
    filepath = 'csv/iEDA_clean_nets.csv'
    # filepath = 'csv/innovus_clean_nets.csv'
    main(filepath)
