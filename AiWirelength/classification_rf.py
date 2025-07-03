import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import classification_report, roc_auc_score

def load_and_prepare_data(filepath, num_bins=7):
    """
    加载数据并准备特征和目标变量，进行等频分箱处理，并计算分箱后每段的均值。
    
    参数：
        filepath (str): 数据文件路径。
        num_bins (int): 划分的类别数(默认为5)。
        
    返回：
        data (DataFrame): 处理后的数据。
        X (DataFrame): 特征数据。
        y (Series): 离散化后的目标变量。
        features (list): 特征名称列表。
        bins (list): 动态生成的区间（分位点）。
        bin_means (dict): 每个分箱的均值。
    """
    # 加载数据
    data = pd.read_csv(filepath)
    
    # 去除 pin_num = 3 的样本
    data = data[data['pin_num'] != 3]
    
    # 计算目标变量
    data['target'] = data['rsmt'] / data['hpwl']

    # 查看目标变量的描述性统计信息
    print(data['target'].describe())
    
    # 根据目标变量分布动态生成等频分箱
    bins = np.quantile(data['target'], q=np.linspace(0, 1, num_bins + 1))  # 动态生成等频分位点
    labels = list(range(num_bins))  # 类别标签
    
    # 将目标变量离散化为多类别
    data['target_bin'] = pd.cut(data['target'], bins=bins, labels=labels, include_lowest=True)
    data['target_bin'] = data['target_bin'].astype(int)  # 转换为整数类型

    # 计算每个分箱的均值、最小值和最大值
    bin_stats = data.groupby('target_bin')['target'].agg(['mean', 'min', 'max']).to_dict(orient='index')

    # 打印每个分箱的统计信息
    print("Bin Statistics:")
    for bin_label, stats in bin_stats.items():
        print(f"Bin {bin_label}: Mean = {stats['mean']:.4f}, Min = {stats['min']:.4f}, Max = {stats['max']:.4f}")
    
    # 选择特征
    features = ['pin_num', 'aspect_ratio', 'bbox_width', 'bbox_height', 
                'bbox_area', 'lx', 'ly', 'ux', 'uy', 'lness', 'hpwl', 
                'x_entropy', 'y_entropy', 'x_avg_nn_dist', 'x_std_nn_dist',
                'x_ratio_nn_dist', 'y_avg_nn_dist', 'y_std_nn_dist', 'y_ratio_nn_dist']
    X = data[features]
    y = data['target_bin']
    
    return data, X, y, bins, features

def visualize_target_distribution(data, bins, output_fig):
    """
    可视化目标变量的分布。
    
    参数：
        data (DataFrame): 包含目标变量的处理后数据。
        bins (list): 动态生成的区间。
        output_fig (str): 输出图片的路径。
    """
    plt.figure(figsize=(12, 6))
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：原始目标变量分布
    ax1.hist(data['target'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Original Target Distribution')
    ax1.set_xlabel('Target Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图：等频分箱后的分布
    bin_counts = data['target_bin'].value_counts().sort_index()
    ax2.bar(range(len(bin_counts)), bin_counts.values, 
            color='blue', edgecolor='black', alpha=0.7)
    
    # 添加每个分箱的区间范围标签
    bin_labels = [f'[{bins[i]:.3f},\n{bins[i+1]:.3f}]' for i in range(len(bins)-1)]
    ax2.set_xticks(range(len(bin_counts)))
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    
    ax2.set_title('Equal Frequency Binning Distribution')
    ax2.set_xlabel('Bin Ranges')
    ax2.set_ylabel('Frequency')
    ax2.grid(axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_fig, bbox_inches='tight', dpi=300)
    plt.close()

def plot_pin_distribution(csv_path, save_path):
    """
    读取CSV文件并绘制柱状图,显示每个pin_num的数量分布。

    参数：
        csv_path (str): CSV文件的路径。
        save_path (str): 保存生成图表的路径。
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    # 去除 pin_num = 3 的样本
    df = df[df['pin_num'] != 3]

    # 统计每个pin_num的数量
    pin_counts = df['pin_num'].value_counts().sort_index()

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 创建柱状图
    bars = plt.bar(pin_counts.index, pin_counts.values, 
                   color='skyblue',  # 设置柱子颜色
                   edgecolor='black',  # 设置边框颜色
                   width=0.6)  # 设置柱子宽度

    # 在每个柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',  # 转换为整数
                 ha='center',  # 水平居中对齐
                 va='bottom')  # 垂直对齐到底部

    # 设置图表属性
    plt.xlabel('Number of Pins')  # x轴标签
    plt.ylabel('Number of Nets')  # y轴标签
    plt.title('Distribution of Nets by Pin Numbers')  # 图表标题

    # 添加网格线
    plt.grid(True, alpha=0.3, axis='y')

    # 设置x轴刻度为整数
    plt.xticks(pin_counts.index)

    # 调整y轴范围，让数据标签更容易看清
    plt.ylim(0, max(pin_counts.values) * 1.1)

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到 {save_path}")


def calculate_correlation(data, features):
    """
    计算特征与原始目标变量和分箱后目标变量之间的相关系数
    
    参数：
        data (DataFrame): 包含原始和分箱后目标变量的数据
        features (list): 特征名称列表
    """
    # 计算与原始目标变量的相关系数
    correlation_original = data[features + ['target']].corr()['target'].drop('target')
    
    # 计算与分箱后目标变量的相关系数
    correlation_binned = data[features + ['target_bin']].corr()['target_bin'].drop('target_bin')
    
    # 创建对比数据框
    comparison = pd.DataFrame({
        'Original Correlation': correlation_original,
        'Binned Correlation': correlation_binned
    })
    
    print("\n特征与目标变量的相关系数对比:")
    print(comparison.sort_values(by='Original Correlation', ascending=False))
    
    return comparison

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    分割数据集并进行标准化
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def calculate_baseline(y_train, y_test):
    """
    计算基准模型性能
    """
    baseline_prediction = y_train.mode()[0]  # 找到训练集中最多的类别
    baseline_predictions = np.full_like(y_test, baseline_prediction)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    print("基准模型性能：")
    print(f"准确率: {baseline_accuracy:.6f}")
    return baseline_predictions, baseline_accuracy

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    训练随机森林分类模型
    """
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n模型训练时间: {training_time:.4f} 秒")
    return rf_model, training_time

def evaluate_model(rf_model, X_test, y_test):
    """
    评估模型性能
    """
    start_time = time.time()
    y_pred = rf_model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time
    
    model_accuracy = accuracy_score(y_test, y_pred)
    print("\n随机森林模型性能")
    print(f"准确率: {model_accuracy:.6f}")
    print(f"模型预测时间: {prediction_time:.4f} 秒")
    return y_pred, model_accuracy, prediction_time

def plot_feature_importance(rf_model, features, output_path):
    """
    可视化特征重要性
    """
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n特征重要性")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('RF - feature_importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_classification_report(y_test, y_pred):
    """
    打印分类报告
    """
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

# 过采样少数类
def oversample_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Before Resampling: {Counter(y_train)}")
    print(f"After Resampling: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def plot_target_distribution(y):
    """
    绘制目标变量的分布
    """
    plt.figure(figsize=(10, 6))
    
    # 使用 seaborn 绘制直方图和核密度估计图
    sns.histplot(y, kde=True, bins=30, color='blue', alpha=0.7)
    
    # 添加标题和坐标轴标签
    plt.title('The distribution of target', fontsize=16)
    plt.xlabel('Target', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # 显示图形
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('fig/classify_target_distribution.png')

def create_engineered_features(data):
    """
    基于物理意义创建新特征
    """
    # 1. 密度特征
    data['pin_density'] = data['pin_num'] / data['bbox_area']  # 单位面积引脚数
    
    # 2. 引脚分布特征
    # 结合lness和引脚数量
    data['weighted_lness'] = data['lness'] * np.log1p(data['pin_num'])
    
    # 3. 边界利用率
    data['perimeter_pin_ratio_x'] = data['bbox_width'] / (data['bbox_area'])
    data['perimeter_pin_ratio_y'] = data['bbox_height']/ (data['bbox_area'])
    
    return data

def main(filepath, output_importance_path, output_prediction_path):
    """
    主函数：整合所有步骤
    """
    # 1. 加载和准备数据
    data, X, y, bins, features = load_and_prepare_data(filepath, num_bins=7)

    # 2. 可视化目标变量分布
    # visualize_target_distribution(data, bins, "fig/classify_target_distribution.png")
    
    # 2. 添加工程特征
    data = create_engineered_features(data)
    
    # 3. 更新特征列表
    engineered_features = [
        'pin_density',
        'weighted_lness',
        'perimeter_pin_ratio_x',
        'perimeter_pin_ratio_y'
    ]
    features.extend(engineered_features)
    
    # 4. 计算并显示所有特征的相关系数
    correlation_results = calculate_correlation(data, features)
    print("特征与目标变量的相关系数对比:")
    print(correlation_results)

    # 4. 分割和标准化数据
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)

    # 5. 过采样少数类
    X_train_resampled, y_train_resampled = oversample_data(X_train_scaled, y_train)

    # 6. 训练随机森林模型（设置 class_weight）
    rf_model, training_time = train_random_forest(X_train_resampled, y_train_resampled)

    # 7. 评估模型
    y_pred, model_accuracy, prediction_time = evaluate_model(rf_model, X_test_scaled, y_test)

    # 8. 可视化特征重要性
    plot_feature_importance(rf_model, features, output_importance_path)

    # 10. 打印分类报告
    print_classification_report(y_test, y_pred)

# 调用主函数
if __name__ == "__main__":
    filepath = "csv/iEDA_clean_nets.csv"
    # filepath = "csv/innovus_clean_nets.csv"
    output_importance_path = 'fig/classify_importance.png'
    output_prediction_path = 'fig/classify_prediction.png'
    main(filepath, output_importance_path, output_prediction_path)
