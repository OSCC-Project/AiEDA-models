import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time  # 导入时间模块
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.utils import resample


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

def weighted_resample(X_train, y_train, n_samples=None, random_state=42, power=2):
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

def calculate_correlation(data, features):
    """
    计算并显示所有特征的相关系数
    """
    correlation = data[features + ['target']].corr()['target'].sort_values(ascending=False)
    print("特征与目标变量的相关系数：")
    print(correlation)


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
    plt.savefig('fig/regression_target_distribution.png')

def plot_residuals(y_true, y_pred, model_name="Model"):
    """
    绘制残差图
    """
    residuals = y_true - y_pred  # 计算残差

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='blue', edgecolor='k')
    plt.axhline(0, color='red', linestyle='--', linewidth=2)  # 添加参考线
    plt.title(f"Residual Plot for {model_name}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(alpha=0.3)
    plt.savefig('fig/regression_residuals.png')

def weighted_mse(y_true, y_pred):
    """
    自定义加权 MSE 损失函数
    """
    residual = y_true - y_pred
    weights = 1 / (y_true + 1)  # 根据目标值加权，目标值越大权重越小
    # weights = y_true + 1  # 根据目标值加权，目标值越大权重越大
    grad = -2 * weights * residual
    hess = 2 * weights
    return grad, hess


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    训练和评估多个回归模型
    """
    # 定义模型集合
    models = {
        # "Linear Regression": LinearRegression(),
        # "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        # "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        # "Support Vector Machine": SVR(),
        # "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42, n_estimators=100, objective="reg:squarederror",
                                    max_depth=7, colsample_bytree =0.8, learning_rate=0.1, 
                                    reg_alpha=0.1, reg_lambda=2, subsample=1.0)
    }

    for model_name, model in models.items():
        print(f"\n训练和评估模型: {model_name}")
        
        # 记录训练时间
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # 记录预测时间
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time

        # 计算性能指标
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        # 打印性能指标
        print(f"模型训练时间: {training_time:.4f} 秒")
        print(f"预测时间: {prediction_time:.4f} 秒")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        # 绘制残差图
        print(f"绘制残差图: {model_name}")
        plot_residuals(y_test, y_pred, model_name=model_name)

def apply_tsne_with_test(X_train, y_train, X_test, y_test, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    
    # 合并训练集和测试集
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    
    # t-SNE 降维
    X_combined_tsne = tsne.fit_transform(X_combined)
    
    # 分离降维后的训练集和测试集
    X_train_tsne = X_combined_tsne[:len(X_train)]
    X_test_tsne = X_combined_tsne[len(X_train):]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 训练集
    scatter_train = ax[0].scatter(
        X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap='viridis', alpha=0.7
    )
    ax[0].set_title("Train Data")
    fig.colorbar(scatter_train, ax=ax[0], label='Target Value')

    # 测试集
    scatter_test = ax[1].scatter(
        X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_test, cmap='viridis', alpha=0.7
    )
    ax[1].set_title("Test Data")
    fig.colorbar(scatter_test, ax=ax[1], label='Target Value')
    
    plt.savefig('fig/regression_tsne_train_test.png')

def optimize_xgboost(X_train, y_train, X_test, y_test):
    """
    使用 GridSearchCV 对 XGBoost 进行超参数调优
    """
    # 定义 XGBoost 模型
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # 定义超参数搜索空间
    param_grid = {
        'n_estimators': [50, 100, 200],  # 树的数量
        'max_depth': [3, 5, 7],          # 树的深度
        'learning_rate': [0.01, 0.1, 0.2],  # 学习率
        'subsample': [0.6, 0.8, 1.0],    # 子采样比例
        'colsample_bytree': [0.6, 0.8, 1.0],  # 特征采样比例
        'reg_alpha': [0, 0.1, 1],        # L1 正则化
        'reg_lambda': [1, 1.5, 2]        # L2 正则化
    }

    # 使用 GridSearchCV 进行超参数搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # 使用负均方误差作为评分标准
        cv=3,  # 3 折交叉验证
        verbose=1,
        n_jobs=-1  # 使用所有可用 CPU 核心
    )

    print("\n开始超参数调优...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    print(f"超参数调优完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳得分: {-grid_search.best_score_:.4f}")

    # 返回最佳模型
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    使用最佳模型对测试集进行评估
    """
    print("\n评估最佳模型...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    # 计算性能指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # 打印性能指标
    print(f"预测时间: {end_time - start_time:.4f} 秒")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")

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


def main(filepath):
    """
    主函数：整合所有步骤
    """
    # 1. 加载和准备数据
    X, y, data, features = load_and_prepare_data(filepath)

    # 2. 创建新的特征
    data = create_engineered_features(data)

    # 3. 更新特征列表
    engineered_features = [
        'std_nn_ratio',
        'route_ratio_x',
        'route_ratio_y'
    ]
    features.extend(engineered_features)
    X = data[features]  # 更新 X

    # 4. 计算并显示所有特征的相关系数
    # calculate_correlation(data, features)

    # 5. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. 对训练集进行重复采样（平衡目标变量分布）
    X_train_resampled, y_train_resampled = weighted_resample(X_train, y_train, n_samples=40000, power=2)

    # 绘制采样前后的目标变量分布
    plot_resampled_target_distribution(y_train, y_train_resampled)

    # 7. 对特征进行归一化（避免数据泄露）
    scaler = MinMaxScaler()
    # scaler = StandardScaler()

    # 在训练集上拟合归一化器
    X_train_resampled = scaler.fit_transform(X_train_resampled)

    # 使用训练集的归一化参数对测试集进行变换
    X_test = scaler.transform(X_test)

    # 8. 可视化训练集和测试集的分布
    # apply_tsne_with_test(X_train_resampled, y_train_resampled, X_test, y_test, n_components=2)

    # 9. 训练和评估多个模型
    train_and_evaluate_models(X_train_resampled, X_test, y_train_resampled, y_test)

    # 10. 超参数调优并评估模型
    # best_model = optimize_xgboost(X_train_resampled, y_train_resampled, X_test, y_test)
    # evaluate_model(best_model, X_test, y_test)

# 调用主函数
if __name__ == "__main__":
    filepath = 'csv/iEDA_clean_nets.csv'
    # filepath = 'csv/innovus_clean_nets.csv'

    main(filepath)
