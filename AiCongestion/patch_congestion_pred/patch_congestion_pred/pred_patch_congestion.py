import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as skimage_ssim
from sklearn.metrics import mean_squared_error
from tqdm import tqdm # 导入 tqdm
from pytorch_msssim import ssim as torch_ssim
# --- 1. 定义常量和路径 ---
TRAIN_DIR = '/home/zhanghongda/AI4EDA/AiCongestion/patch_congestion_pred/total_fea/train'
TEST_DIR = '/home/zhanghongda/AI4EDA/AiCongestion/patch_congestion_pred/total_fea/test'
# 指定的输出目录，所有生成的图、模型、数据都将保存到这里
OUTPUT_DIR = '/home/zhanghongda/AI4EDA/AiCongestion/patch_congestion_pred/patch_congestion_pred/haokanCNN'

FEATURE_COLUMNS = [
    'area', 'cell_density', 'pin_density',
    'net_density', 'RUDY_congestion',
    'timing', 'power'
] 
TARGET_COLUMN = 'EGR_congestion'
PATCH_SIZE = 8 # 定义滑动窗口的大小

# --- 配置设备（GPU/CPU） ---
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available. Using {torch.cuda.device_count()} GPU(s).")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")
    return device

# --- 2. 数据加载和预处理工具函数 ---

def load_and_process_csv(filepath, feature_cols, target_col, scaler=None, is_training=True):
    """
    加载单个CSV文件，并进行预处理。
    Args:
        filepath (str): CSV文件路径。
        feature_cols (list): 输入特征列名列表。
        target_col (str): 目标特征列名。
        scaler (StandardScaler): 用于特征缩放的StandardScaler对象。
        is_training (bool): 是否为训练模式，用于判断是否fit scaler。
    Returns:
        tuple: (df, design_name) 处理后的DataFrame和设计名称。
    """
    df = pd.read_csv(filepath)
    design_name = os.path.basename(filepath).split('_archinfo_patch_features_new_total.csv')[0]

    # 特征缩放
    if scaler:
        if is_training:
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = scaler.transform(df[feature_cols])

    return df, design_name

def create_full_image_from_dataframe(df, feature_cols, target_col):
    """
    从DataFrame创建整个设计的图像表示。
    Args:
        df (pd.DataFrame): 包含patch数据的DataFrame。
        feature_cols (list): 输入特征列名列表。
        target_col (str): 目标特征列名。
    Returns:
        tuple: (feature_full_image, target_full_image) 整个设计的特征图像和目标图像的numpy数组。
               feature_full_image shape: (num_channels, max_row+1, max_col+1)
               target_full_image shape: (max_row+1, max_col+1)
    """
    max_row = df['patch_id_row'].max()
    max_col = df['patch_id_col'].max()

    num_channels = len(feature_cols)
    feature_full_image = np.zeros((num_channels, max_row + 1, max_col + 1), dtype=np.float32)
    target_full_image = np.zeros((max_row + 1, max_col + 1), dtype=np.float32)

    for _, row in df.iterrows():
        r = int(row['patch_id_row'])
        c = int(row['patch_id_col'])
        feature_full_image[:, r, c] = row[feature_cols].values
        target_full_image[r, c] = row[target_col]

    return feature_full_image, target_full_image

def extract_patches(full_feature_image, full_target_image, patch_size, stride=1):
    """
    从完整的特征图像和目标图像中提取8*8的patches。
    仅提取完全位于图像内部的样本，不进行零填充。
    Args:
        full_feature_image (np.array): 完整的特征图像 (C, H, W)。
        full_target_image (np.array): 完整的目标图像 (H, W)。
        patch_size (int): patch的边长。
        stride (int): 滑动步长。
    Returns:
        list: 包含 (feature_patch, target_patch, original_coords) 的列表。
              original_coords: (row_start, col_start)
    """
    patches = []
    num_channels, H, W = full_feature_image.shape

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            feature_patch = full_feature_image[:, r:r+patch_size, c:c+patch_size]
            target_patch = full_target_image[r:r+patch_size, c:c+patch_size]
            patches.append((feature_patch, target_patch, (r, c)))
    return patches

# --- 3. 自定义数据集类 ---

class CongestionDataset(Dataset):
    def __init__(self, data_dir, feature_cols, target_col, patch_size, scaler=None, is_training=True):
        self.data_dir = data_dir
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.patch_size = patch_size
        self.scaler = scaler
        self.is_training = is_training
        
        self.design_full_dims = {} 
        self.samples = self._load_and_extract_patches()

    def _load_and_extract_patches(self):
        all_patches = []
        filepaths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        if self.is_training and self.scaler is None:
            all_dfs = []
            for fp in filepaths:
                df, _ = pd.read_csv(fp), os.path.basename(fp).split('_archinfo_patch_features_new_total.csv')[0]
                all_dfs.append(df)
            combined_df = pd.concat(all_dfs)
            self.scaler = StandardScaler()
            self.scaler.fit(combined_df[self.feature_cols])
            print("StandardScaler fitted on training features.")
        elif not self.is_training and self.scaler is None:
            raise ValueError("For evaluation (is_training=False), a pre-fitted scaler must be provided.")

        for filepath in filepaths:
            df, design_name = load_and_process_csv(filepath, self.feature_cols, self.target_col, self.scaler, self.is_training)
            full_feature_img, full_target_img = create_full_image_from_dataframe(df, self.feature_cols, self.target_col)
            
            self.design_full_dims[design_name] = (full_feature_img.shape[1], full_feature_img.shape[2]) # (H, W)

            patches_for_design = extract_patches(full_feature_img, full_target_img, self.patch_size)
            
            for feature_patch, target_patch, coords in patches_for_design:
                all_patches.append((feature_patch, target_patch, design_name, coords))
        return all_patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_patch, target_patch, design_name, coords = self.samples[idx]
        if np.isnan(feature_patch[6, :, :]).any():
            feature_patch[6, :, :] = np.nan_to_num(feature_patch[6, :, :], nan=0.0)        
        return torch.tensor(feature_patch, dtype=torch.float32), \
               torch.tensor(target_patch, dtype=torch.float32), \
               design_name, coords

class CongestionCNN(nn.Module):
    def __init__(self, in_channels):
        super(CongestionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.2),  # 防止过拟合

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x
# --- 5. 评估指标函数 ---

def calculate_nrmse(true_map, predicted_map):
    """计算归一化均方根误差 (Normalized Root Mean Squared Error)。"""
    rmse = np.sqrt(mean_squared_error(true_map.flatten(), predicted_map.flatten()))
    data_range = np.max(true_map) - np.min(true_map)
    if data_range == 0:
        return 0.0
    return rmse / data_range

def calculate_skimage_ssim(true_map, predicted_map):
    """计算结构相似性指数 (Structural Similarity Index Measure)。"""
    min_val = min(np.min(true_map), np.min(predicted_map))
    max_val = max(np.max(true_map), np.max(predicted_map))

    if max_val == min_val:
        return 1.0

    true_map_norm = (true_map - min_val) / (max_val - min_val)
    predicted_map_norm = (predicted_map - min_val) / (max_val - min_val)

    return skimage_ssim(true_map_norm, predicted_map_norm, data_range=1.0)


# --- 6. 训练和评估函数 ---

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型。
    Args:
        model (nn.Module): 待训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        num_epochs (int): 训练的epoch数量。
        device (torch.device): 运行设备 (CPU/GPU)。
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        # 使用 tqdm 包装 train_loader，显示进度条
        # desc 参数设置进度条的描述
        # leave=True 表示在循环结束后保留进度条
        for inputs, targets, design_names, coords in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            targets = targets.unsqueeze(1) 
            '''
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)'''
            alpha = 0.8  # MSE和SSIM的权重，可根据实际调整

            # 归一化到[0,1]，防止SSIM数值不稳定
            t_min = targets.min()
            t_max = targets.max()
            targets_norm = (targets - t_min) / (t_max - t_min + 1e-8)
            outputs_norm = (outputs - t_min) / (t_max - t_min + 1e-8)

            mse_loss = criterion(outputs, targets)
            ssim_loss = 1 - torch_ssim(outputs_norm, targets_norm, data_range=1.0, size_average=True)
            loss = alpha * mse_loss + (1 - alpha) * ssim_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader.dataset):.6f}")

def evaluate_model(model, test_loader, criterion, design_full_dims, device):
    """
    评估模型并重建完整预测图。
    """
    model.eval()
    total_loss = 0.0
    all_design_data = {} 

    with torch.no_grad():
        # 评估过程也可以加进度条，例如：
        for inputs, targets, design_names, coords in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            targets_unsqueezed = targets.unsqueeze(1)
            
            loss = criterion(outputs, targets_unsqueezed)
            total_loss += loss.item() * inputs.size(0)

            for i in range(inputs.size(0)):
                design_name = design_names[i]
                r_start, c_start = coords[0][i].item(), coords[1][i].item()
                
                original_H, original_W = design_full_dims[design_name]

                if design_name not in all_design_data:
                    all_design_data[design_name] = {
                        'predicted_map': np.zeros((original_H, original_W), dtype=np.float32),
                        'true_map': np.zeros((original_H, original_W), dtype=np.float32),
                        'patch_count_map': np.zeros((original_H, original_W), dtype=np.int32),
                    }
                
                design_info = all_design_data[design_name]
                
                pred_patch = outputs[i].squeeze().cpu().numpy()
                true_patch = targets[i].squeeze().cpu().numpy()
                
                design_info['predicted_map'][r_start:r_start+PATCH_SIZE, c_start:c_start+PATCH_SIZE] += pred_patch
                design_info['true_map'][r_start:r_start+PATCH_SIZE, c_start:c_start+PATCH_SIZE] += true_patch
                design_info['patch_count_map'][r_start:r_start+PATCH_SIZE, c_start:c_start+PATCH_SIZE] += 1
    
    final_results = {}
    for design_name, info in all_design_data.items():
        if np.all(info['patch_count_map'] == 0):
            print(f"Warning: Design {design_name} has no valid patches for reconstruction. Skipping.")
            continue

        nonzero_mask = info['patch_count_map'] > 0
        
        if not np.any(nonzero_mask):
            print(f"Skipping {design_name}: True map contains only zeros after masking. Cannot calculate skimage_ssim/NRMSE.")
            continue

        avg_pred_map = np.zeros_like(info['predicted_map'])
        avg_pred_map[nonzero_mask] = info['predicted_map'][nonzero_mask] / info['patch_count_map'][nonzero_mask]
        
        avg_true_map = np.zeros_like(info['true_map'])
        avg_true_map[nonzero_mask] = info['true_map'][nonzero_mask] / info['patch_count_map'][nonzero_mask]
        
        final_results[design_name] = {
            'predicted': avg_pred_map,
            'true': avg_true_map
        }

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.6f}")
    return final_results

# --- 7. 可视化函数 (用于测试集) ---

def plot_congestion_map_with_metrics(predicted_map, true_map, design_name, skimage_ssim_score, nrmse_score):
    """
    绘制预测拥塞图和真实拥塞图，并显示skimage_ssim和NRMSE。
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sns.heatmap(true_map, cmap='viridis', annot=False, cbar_kws={'label': 'EGR Congestion'}, ax=axes[0])
    axes[0].set_title(f"True EGR Congestion - {design_name}")
    axes[0].set_xlabel("Patch Column ID")
    axes[0].set_ylabel("Patch Row ID")

    sns.heatmap(predicted_map, cmap='viridis', annot=False, cbar_kws={'label': 'EGR Congestion'}, ax=axes[1])
    axes[1].set_title(f"Predicted EGR Congestion - {design_name}\nskimage_ssim: {skimage_ssim_score:.4f}, NRMSE: {nrmse_score:.4f}")
    axes[1].set_xlabel("Patch Column ID")
    axes[1].set_ylabel("Patch Row ID")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{design_name}_congestion_comparison.png"))
    plt.close()

# --- 8. 主执行流程 ---

if __name__ == '__main__':
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 获取运行设备
    device = get_device()
    num_gpus = torch.cuda.device_count()

    # 设置数据加载器的工作进程数
    num_workers = min(os.cpu_count(), 8) if os.cpu_count() else 0
    print(f"Using {num_workers} workers for DataLoader.")

    # 训练集和测试集共享同一个 scaler
    train_dataset = CongestionDataset(TRAIN_DIR, FEATURE_COLUMNS, TARGET_COLUMN, PATCH_SIZE, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)

    test_dataset = CongestionDataset(TEST_DIR, FEATURE_COLUMNS, TARGET_COLUMN, PATCH_SIZE, scaler=train_dataset.scaler, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    model = CongestionCNN(in_channels=len(FEATURE_COLUMNS))
    
    # 如果有多个GPU，使用nn.DataParallel包装模型
    if num_gpus > 1:
        print(f"Wrapping model with nn.DataParallel to utilize {num_gpus} GPUs.")
        model = nn.DataParallel(model)
    
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=20, device=device)
    print("Training finished.")

    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    model_save_path = os.path.join(OUTPUT_DIR, "congestion_cnn_model.pth")
    torch.save(model_state_dict, model_save_path)
    print(f"Trained model saved to {model_save_path}")

    print("Evaluating model on test set and calculating metrics...")
    test_results = evaluate_model(model, test_loader, criterion, test_dataset.design_full_dims, device=device)
    print("Evaluation finished.")

    print("\n--- Evaluation Metrics and Visualizations ---")
    
    all_skimage_ssims = []
    all_nrmses = []
    design_metrics = {}

    test_design_names = sorted(list(test_dataset.design_full_dims.keys()))

    for design_name in test_design_names:
        if design_name not in test_results:
            print(f"Warning: Design {design_name} was not fully processed or had no valid patches in test_results. Skipping metric calculation and plotting.")
            continue

        maps_dict = test_results[design_name]
        predicted_map = maps_dict['predicted']
        true_map = maps_dict['true']

        if not np.any(true_map) or true_map.shape != predicted_map.shape:
            print(f"Skipping {design_name}: True map is empty, contains only zeros, or dimensions mismatch. Cannot calculate skimage_ssim/NRMSE.")
            plot_congestion_map_with_metrics(predicted_map, true_map, design_name, float('nan'), float('nan'))
            continue

        valid_mask = true_map != 0 
        
        if not np.any(valid_mask):
            print(f"Skipping {design_name}: True map contains only zeros after masking. Cannot calculate skimage_ssim/NRMSE.")
            plot_congestion_map_with_metrics(predicted_map, true_map, design_name, float('nan'), float('nan'))
            continue

        skimage_ssim_score = calculate_skimage_ssim(true_map[valid_mask], predicted_map[valid_mask])
        nrmse_score = calculate_nrmse(true_map[valid_mask], predicted_map[valid_mask])
        
        all_skimage_ssims.append(skimage_ssim_score)
        all_nrmses.append(nrmse_score)
        design_metrics[design_name] = {'skimage_ssim': skimage_ssim_score, 'NRMSE': nrmse_score}

        print(f"Design: {design_name}, skimage_ssim: {skimage_ssim_score:.4f}, NRMSE: {nrmse_score:.4f}")
        
        np.savez(os.path.join(OUTPUT_DIR, f"{design_name}_congestion_data.npz"),
                 predicted_map=predicted_map,
                 true_map=true_map,
                 skimage_ssim_score=skimage_ssim_score,
                 nrmse_score=nrmse_score)

        plot_congestion_map_with_metrics(predicted_map, true_map, design_name, skimage_ssim_score, nrmse_score)
    
    if all_skimage_ssims and all_nrmses:
        avg_skimage_ssim = np.mean(all_skimage_ssims)
        avg_nrmse = np.mean(all_nrmses)

        average_metrics_path = os.path.join(OUTPUT_DIR, "average_metrics.txt")
        with open(average_metrics_path, "w") as f:
            f.write(f"Average skimage_ssim across all valid test designs: {avg_skimage_ssim:.4f}\n")
            f.write(f"Average NRMSE across all valid test designs: {avg_nrmse:.4f}\n")
            f.write("\nIndividual Design Metrics:\n")
            for d_name in sorted(design_metrics.keys()):
                metrics = design_metrics[d_name]
                f.write(f"  {d_name}: skimage_ssim={metrics['skimage_ssim']:.4f}, NRMSE={metrics['NRMSE']:.4f}\n")

        print(f"\n--- Overall Average Metrics ---")
        print(f"Average skimage_ssim across all valid test designs: {avg_skimage_ssim:.4f}")
        print(f"Average NRMSE across all valid test designs: {avg_nrmse:.4f}")
        print(f"Detailed metrics and average metrics saved to {average_metrics_path}")
    else:
        print("\nNo valid test designs to calculate average metrics.")