import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def plot_distributions(df_ieda, df_innovus, output_path):
    """
    绘制 iEDA 和 Innovus 的 R(P)/B(P) 分布图，并保存为图片。

    参数:
    - df_ieda: iEDA 的数据 DataFrame。
    - df_innovus: Innovus 的数据 DataFrame。
    - output_path: 输出图像的保存路径。
    """
    def plot_distribution(ax, df, title, colors):
        """
        在给定的子图上绘制 R(P)/B(P) 的分布图。
        """
        # 排除 pin_num=3 的数据
        df = df[df['pin_num'] != 3]

        pin_nums = sorted(df['pin_num'].unique())
        x_eval = np.linspace(0, 1, 200)

        for pin_num, color in zip(pin_nums, colors):
            # 获取特定 pin_num 的数据
            subset = df[df['pin_num'] == pin_num]
            total_nets = len(subset)  # 该 pin_num 对应的总 net 数

            # 使用高斯核密度估计
            kde = stats.gaussian_kde(subset['lness'], bw_method=0.1)

            # 计算密度并转换为百分比
            density = kde(x_eval)
            # 归一化处理：确保总面积为1，然后转换为百分比
            density = density / np.trapz(density, x_eval)

            ax.plot(x_eval, density, label=f'p={pin_num} (n={total_nets})', color=color, linewidth=2)

        ax.set_xlabel('R(P)/B(P)')
        ax.set_ylabel('Percentage of nets (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_facecolor('#F0F0F8')

    # 设置颜色
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'pink']

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)  # 一行两列子图，y轴共享
    plt.style.use('seaborn')

    # 绘制 iEDA 数据分布
    plot_distribution(axes[0], df_ieda, 'iEDA: Distribution of R(P)/B(P)', colors)

    # 绘制 Innovus 数据分布
    plot_distribution(axes[1], df_innovus, 'Innovus: Distribution of R(P)/B(P)', colors)

    # 设置整体标题
    fig.suptitle('Comparison of R(P)/B(P) Distributions for iEDA and Innovus', fontsize=16)

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_distributions_same_pin(df_ieda, df_innovus, output_path):
    """
    绘制 iEDA 和 Innovus 的 R(P)/B(P) 分布图，并保存为图片。
    确保相同pin_num下两组数据具有相同的样本数。

    参数:
    - df_ieda: iEDA 的数据 DataFrame。
    - df_innovus: Innovus 的数据 DataFrame。
    - output_path: 输出图像的保存路径。
    """
    def balance_samples(df_ieda, df_innovus):
        """
        确保两个DataFrame在相同pin_num下具有相同的样本数。
        返回处理后的两个DataFrame。
        """
        balanced_ieda_data = []
        balanced_innovus_data = []
        
        # 获取共同的pin_num
        common_pins = sorted(set(df_ieda['pin_num'].unique()) & set(df_innovus['pin_num'].unique()))
        common_pins = [p for p in common_pins if p != 3]  # 排除pin_num=3
        
        for pin_num in common_pins:
            ieda_group = df_ieda[df_ieda['pin_num'] == pin_num]
            innovus_group = df_innovus[df_innovus['pin_num'] == pin_num]
            
            # 确定最小样本数
            min_samples = min(len(ieda_group), len(innovus_group))
            
            # 随机采样到相同样本数
            if len(ieda_group) > min_samples:
                ieda_group = ieda_group.sample(n=min_samples, random_state=42)
            if len(innovus_group) > min_samples:
                innovus_group = innovus_group.sample(n=min_samples, random_state=42)
            
            balanced_ieda_data.append(ieda_group)
            balanced_innovus_data.append(innovus_group)
        
        return pd.concat(balanced_ieda_data), pd.concat(balanced_innovus_data)

    def plot_distribution(ax, df, title, colors):
        """
        在给定的子图上绘制 R(P)/B(P) 的分布图。
        """
        pin_nums = sorted(df['pin_num'].unique())
        x_eval = np.linspace(0, 1, 200)

        for pin_num, color in zip(pin_nums, colors):
            # 获取特定 pin_num 的数据
            subset = df[df['pin_num'] == pin_num]
            total_nets = len(subset)  # 该 pin_num 对应的总 net 数

            # 使用高斯核密度估计
            kde = stats.gaussian_kde(subset['lness'], bw_method=0.1)

            # 计算密度并转换为百分比
            density = kde(x_eval)
            # 归一化处理：确保总面积为1，然后转换为百分比
            density = density / np.trapz(density, x_eval)

            ax.plot(x_eval, density, label=f'p={pin_num} (n={total_nets})', color=color, linewidth=2)

        ax.set_xlabel('R(P)/B(P)')
        ax.set_ylabel('Percentage of nets (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_facecolor('#F0F0F8')

    # 首先平衡两组数据的样本数
    balanced_df_ieda, balanced_df_innovus = balance_samples(df_ieda, df_innovus)

    # 设置颜色
    # colors = ['blue', 'orange', 'green', 'red', 'purple', 'pink']
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)  # 一行两列子图，y轴共享
    plt.style.use('seaborn')

    # 绘制 iEDA 数据分布
    plot_distribution(axes[0], balanced_df_ieda, 'iEDA: Distribution of R(P)/B(P)', colors)

    # 绘制 Innovus 数据分布
    plot_distribution(axes[1], balanced_df_innovus, 'Innovus: Distribution of R(P)/B(P)', colors)

    # 设置整体标题
    fig.suptitle('Comparison of R(P)/B(P) Distributions for iEDA and Innovus\n(Balanced samples)', fontsize=16)

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def perform_statistical_tests(df_ieda, df_innovus, output_csv_path):
    """
    对 iEDA 和 Innovus 的数据进行统计检验 (K-S 检验和 t 检验)。

    参数:
    - df_ieda: iEDA 的数据 DataFrame。
    - df_innovus: Innovus 的数据 DataFrame。
    - output_csv_path: 输出统计检验结果的 CSV 文件路径。

    返回:
    - results_df: 包含统计检验结果的 DataFrame。
    """
    # 排除 pin_num=3 的数据
    df_ieda = df_ieda[df_ieda['pin_num'] != 3]
    df_innovus = df_innovus[df_innovus['pin_num'] != 3]

    # 获取所有的 pin_num 值
    pin_nums = sorted(set(df_ieda['pin_num']).intersection(set(df_innovus['pin_num'])))

    # 初始化结果存储
    results = []

    # 对每个 pin_num 分别进行 K-S 检验和 t 检验
    for pin_num in pin_nums:
        # 筛选出当前 pin_num 的数据
        ieda_values = df_ieda[df_ieda['pin_num'] == pin_num]['lness']
        innovus_values = df_innovus[df_innovus['pin_num'] == pin_num]['lness']

        # K-S 检验
        ks_stat, ks_p_value = stats.ks_2samp(ieda_values, innovus_values)

        # t 检验
        t_stat, t_p_value = stats.ttest_ind(ieda_values, innovus_values, equal_var=False)  # 假定方差不相等

        # 存储结果
        results.append({
            'pin_num': pin_num,
            'ks_stat': ks_stat,
            'ks_p_value': ks_p_value,
            't_stat': t_stat,
            't_p_value': t_p_value
        })

    # 将结果转为 DataFrame 以便查看
    results_df = pd.DataFrame(results)

    # 保存结果到 CSV 文件
    results_df.to_csv(output_csv_path, index=False)

    return results_df

def bootstrap_ci(data, num_bootstrap=10000, ci=0.95):
    """
    使用手动实现的 Bootstrapping 方法计算均值的置信区间
    参数:
        data: 原始数据 (1D array)
        num_bootstrap: Bootstrap 抽样次数 (默认 10000)
        ci: 置信水平 (默认 0.95)
    返回:
        (lower_bound, upper_bound): 置信区间的下限和上限
    """
    # 生成 num_bootstrap 个 Bootstrap 样本的均值
    bootstrap_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(num_bootstrap)
    ])
    # 计算置信区间
    lower_bound = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

def plot_mean_trend_with_bootstrap_ci(df_ieda, df_innovus, output_fig):
    """
    绘制 iEDA 和 Innovus 的 R(P)/B(P) 平均值趋势图，包含 iEDA 和 Innovus 的 95% 置信区间 (基于 Bootstrapping)
    """
    # 初始化存储 iEDA 结果的列表
    mean_lness_ieda = []
    ci_upper_ieda = []
    ci_lower_ieda = []

    # 按 pin_num 分组计算 iEDA 的均值和置信区间
    grouped_ieda = df_ieda.groupby('pin_num')['lness']
    for pin_num, group in grouped_ieda:
        mean_value = group.mean()  # 计算均值
        ci_lower, ci_upper = bootstrap_ci(group.values)  # 计算 95% 置信区间
        mean_lness_ieda.append((pin_num, mean_value, ci_lower, ci_upper))

    # 转换为 DataFrame
    mean_lness_ieda = pd.DataFrame(mean_lness_ieda, columns=['pin_num', 'lness', 'ci_lower', 'ci_upper'])

    # 初始化存储 Innovus 结果的列表
    mean_lness_innovus = []
    ci_upper_innovus = []
    ci_lower_innovus = []

    # 按 pin_num 分组计算 Innovus 的均值和置信区间
    grouped_innovus = df_innovus.groupby('pin_num')['lness']
    for pin_num, group in grouped_innovus:
        mean_value = group.mean()  # 计算均值
        ci_lower, ci_upper = bootstrap_ci(group.values)  # 计算 95% 置信区间
        mean_lness_innovus.append((pin_num, mean_value, ci_lower, ci_upper))

    # 转换为 DataFrame
    mean_lness_innovus = pd.DataFrame(mean_lness_innovus, columns=['pin_num', 'lness', 'ci_lower', 'ci_upper'])

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn')

    # 绘制 iEDA 的平均值曲线
    plt.plot(mean_lness_ieda['pin_num'], mean_lness_ieda['lness'], 
             marker='o', 
             color='purple', 
             label='iEDA Mean R(P)/B(P)', 
             linewidth=2)

    # 使用 fill_between 绘制 iEDA 的置信区间区域
    plt.fill_between(mean_lness_ieda['pin_num'], 
                     mean_lness_ieda['ci_lower'], 
                     mean_lness_ieda['ci_upper'], 
                     color='purple', 
                     alpha=0.2, 
                     label='iEDA 95% CI')

    # 绘制 Innovus 的平均值曲线
    plt.plot(mean_lness_innovus['pin_num'], mean_lness_innovus['lness'], 
             marker='s', 
             color='orange', 
             label='Innovus Mean R(P)/B(P)', 
             linewidth=2)

    # 使用 fill_between 绘制 Innovus 的置信区间区域
    plt.fill_between(mean_lness_innovus['pin_num'], 
                     mean_lness_innovus['ci_lower'], 
                     mean_lness_innovus['ci_upper'], 
                     color='orange', 
                     alpha=0.3, 
                     label='Innovus 95% CI')

    # 设置坐标轴
    plt.xlabel('p')
    plt.ylabel('R(P)/B(P)')
    plt.title('Mean R(P)/B(P) Trend with 95% CI for iEDA and Innovus')

    # 设置 x 轴范围和刻度
    # plt.xlim(2.5, 15.5)  # 比原范围稍微扩大
    # plt.xticks(range(3, 16))  # x 轴刻度覆盖完整范围
    plt.xlim(2.5, 31.5)  # 比原范围稍微扩大
    plt.xticks(range(3, 32))  # x 轴刻度覆盖完整范围

    # 设置 y 轴范围，自动调整边界
    y_min = min(mean_lness_ieda['ci_lower'].min(), mean_lness_innovus['ci_lower'].min())
    y_max = max(mean_lness_ieda['ci_upper'].max(), mean_lness_innovus['ci_upper'].max())
    plt.ylim(y_min - 0.05, y_max + 0.05)  # 在最小值和最大值基础上留出 5% 的空间

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 添加图例
    plt.legend()

    # 设置背景颜色
    plt.gca().set_facecolor('#F0F0F8')
    plt.gcf().set_facecolor('white')

    # 自动调整边距，确保边界点显示完整
    plt.margins(x=0.02, y=0.02)  # x 和 y 方向各留出 2% 的边距

    # 保存图表
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mean_trend_with_bootstrap_ci_same_pin(df_ieda, df_innovus, output_fig):
    """
    绘制 iEDA 和 Innovus 的 R(P)/B(P) 平均值趋势图，包含 iEDA 和 Innovus 的 95% 置信区间 (基于 Bootstrapping)
    在相同pin_num下确保两组数据具有相同的样本数,并在两组数据点上都标注样本数
    """
    # 初始化存储结果的列表
    mean_lness_ieda = []
    mean_lness_innovus = []
    sample_sizes = []  # 存储每个pin_num的样本数

    # 获取所有唯一的pin_num值
    all_pin_nums = sorted(set(df_ieda['pin_num'].unique()) & set(df_innovus['pin_num'].unique()))
    
    # 对每个pin_num进行处理
    for pin_num in all_pin_nums:
        # 获取当前pin_num的两组数据
        ieda_group = df_ieda[df_ieda['pin_num'] == pin_num]['lness']
        innovus_group = df_innovus[df_innovus['pin_num'] == pin_num]['lness']
        
        # 确定最小样本数
        min_samples = min(len(ieda_group), len(innovus_group))
        sample_sizes.append(min_samples)  # 记录样本数
        
        # 如果innovus样本数较多，随机采样到与ieda相同
        if len(innovus_group) > min_samples:
            innovus_group = innovus_group.sample(n=min_samples, random_state=42)
        # 如果ieda样本数较多，随机采样到与innovus相同
        elif len(ieda_group) > min_samples:
            ieda_group = ieda_group.sample(n=min_samples, random_state=42)
            
        # 计算iEDA的均值和置信区间
        ieda_mean = ieda_group.mean()
        ieda_ci_lower, ieda_ci_upper = bootstrap_ci(ieda_group.values)
        mean_lness_ieda.append((pin_num, ieda_mean, ieda_ci_lower, ieda_ci_upper))
        
        # 计算Innovus的均值和置信区间
        innovus_mean = innovus_group.mean()
        innovus_ci_lower, innovus_ci_upper = bootstrap_ci(innovus_group.values)
        mean_lness_innovus.append((pin_num, innovus_mean, innovus_ci_lower, innovus_ci_upper))

    # 转换为DataFrame
    mean_lness_ieda = pd.DataFrame(mean_lness_ieda, 
                                 columns=['pin_num', 'lness', 'ci_lower', 'ci_upper'])
    mean_lness_innovus = pd.DataFrame(mean_lness_innovus, 
                                    columns=['pin_num', 'lness', 'ci_lower', 'ci_upper'])

    # 创建图表
    plt.figure(figsize=(12, 7))  # 稍微加大图表尺寸以适应标注
    plt.style.use('seaborn')

    # 绘制 iEDA 的平均值曲线
    plt.plot(mean_lness_ieda['pin_num'], mean_lness_ieda['lness'], 
             marker='o', 
             color='purple', 
             label='iEDA Mean R(P)/B(P)', 
             linewidth=2)

    # 使用 fill_between 绘制 iEDA 的置信区间区域
    plt.fill_between(mean_lness_ieda['pin_num'], 
                     mean_lness_ieda['ci_lower'], 
                     mean_lness_ieda['ci_upper'], 
                     color='purple', 
                     alpha=0.2, 
                     label='iEDA 95% CI')

    # 绘制 Innovus 的平均值曲线
    plt.plot(mean_lness_innovus['pin_num'], mean_lness_innovus['lness'], 
             marker='s', 
             color='orange', 
             label='Innovus Mean R(P)/B(P)', 
             linewidth=2)

    # 使用 fill_between 绘制 Innovus 的置信区间区域
    plt.fill_between(mean_lness_innovus['pin_num'], 
                     mean_lness_innovus['ci_lower'], 
                     mean_lness_innovus['ci_upper'], 
                     color='orange', 
                     alpha=0.3, 
                     label='Innovus 95% CI')

    # 添加样本数标注
    for i, (pin_num, sample_size) in enumerate(zip(all_pin_nums, sample_sizes)):
        # 在iEDA点上方标注
        plt.annotate(f'n={sample_size}', 
                    xy=(pin_num, mean_lness_ieda.iloc[i]['lness']),
                    xytext=(0, -15),  # 向上偏移10个点
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    color='purple',
                    fontsize=8)
        
        # 在Innovus点下方标注
        plt.annotate(f'n={sample_size}', 
                    xy=(pin_num, mean_lness_innovus.iloc[i]['lness']),
                    xytext=(0, 15),  # 向上偏移15个点
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    color='orange',
                    fontsize=8)

    # 设置坐标轴
    plt.xlabel('p')
    plt.ylabel('R(P)/B(P)')
    plt.title('Mean R(P)/B(P) Trend with 95% CI for iEDA and Innovus\n(n=sample size)')

    plt.xlim(2.5, 31.5)  
    plt.xticks(range(3, 32))  

    # 设置 y 轴范围，自动调整边界（考虑标注空间）
    y_min = min(mean_lness_ieda['ci_lower'].min(), mean_lness_innovus['ci_lower'].min())
    y_max = max(mean_lness_ieda['ci_upper'].max(), mean_lness_innovus['ci_upper'].max())
    margin = (y_max - y_min) * 0.15  # 使用相对边距
    plt.ylim(y_min - margin, y_max + margin)  # 上下都留出空间用于标注

    plt.grid(True, alpha=0.3)
    plt.legend()

    # 设置背景颜色
    plt.gca().set_facecolor('#F0F0F8')
    plt.gcf().set_facecolor('white')

    # 自动调整边距
    plt.margins(x=0.02, y=0.05)  # 增加垂直边距以适应标注

    # 保存图表
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()



def plot_wirelength_feature_distribution(df_ieda, df_innovus, output_fig):
    """
    绘制 iEDA 和 Innovus 的 WL 特性分布对比图
    参数:
        df_ieda: 包含 iEDA 数据的 DataFrame
        df_innovus: 包含 Innovus 数据的 DataFrame
        output_fig: 输出图像文件路径
    """
    # 定义目标 lness 值和其他参数
    target_lness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tolerance = 0.045
    pin_nums = [4, 5, 7]
    aspect_ratios = [1, 2, 4]

    # 创建三行三列的子图
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    plt.style.use('seaborn')

    # 设置整个图表的背景色
    plt.gcf().set_facecolor('white')

    # 对每个 aspect_ratio 和 pin_num 组合绘制子图
    for ar_idx, ar in enumerate(aspect_ratios):
        for pin_idx, pin_num in enumerate(pin_nums):
            # 筛选特定 pin_num 和 aspect_ratio 的数据
            df_pin_ieda = df_ieda[(df_ieda['pin_num'] == pin_num) & (df_ieda['aspect_ratio'] == ar)]
            df_pin_innovus = df_innovus[(df_innovus['pin_num'] == pin_num) & (df_innovus['aspect_ratio'] == ar)]
            
            # 创建空的 DataFrame 来存储结果
            results_ieda = []
            results_innovus = []
            
            # 对每个目标 lness 值进行数据筛选和汇总
            for target in target_lness:
                # iEDA 数据
                mask_ieda = (df_pin_ieda['lness'] >= target - tolerance) & (df_pin_ieda['lness'] < target + tolerance)
                subset_ieda = df_pin_ieda[mask_ieda]
                sums_ieda = subset_ieda[['hpwl', 'rsmt', 'grwl']].mean()
                results_ieda.append({
                    'lness': target,
                    'hpwl': sums_ieda['hpwl'],
                    'rsmt': sums_ieda['rsmt'],
                    'grwl': sums_ieda['grwl']
                })
                
                # Innovus 数据
                mask_innovus = (df_pin_innovus['lness'] >= target - tolerance) & (df_pin_innovus['lness'] < target + tolerance)
                subset_innovus = df_pin_innovus[mask_innovus]
                sums_innovus = subset_innovus[['hpwl', 'rsmt', 'grwl']].mean()
                results_innovus.append({
                    'lness': target,
                    'hpwl': sums_innovus['hpwl'],
                    'rsmt': sums_innovus['rsmt'],
                    'grwl': sums_innovus['grwl']
                })
            
            # 转换结果为 DataFrame
            grouped_ieda = pd.DataFrame(results_ieda)
            grouped_innovus = pd.DataFrame(results_innovus)
            
            # 设置子图的背景色
            axes[ar_idx, pin_idx].set_facecolor('#F0F0F8')
            
            # 在对应的子图上绘制 iEDA 的曲线
            axes[ar_idx, pin_idx].plot(grouped_ieda['lness'], grouped_ieda['hpwl'], 'o-', label='iEDA HPWL', linewidth=2, markersize=6)
            axes[ar_idx, pin_idx].plot(grouped_ieda['lness'], grouped_ieda['rsmt'], 's-', label='iEDA RSMT', linewidth=2, markersize=6)
            # axes[ar_idx, pin_idx].plot(grouped_ieda['lness'], grouped_ieda['grwl'], '^-', label='iEDA GRWL', linewidth=2, markersize=6)
            
            # 在对应的子图上绘制 Innovus 的曲线
            axes[ar_idx, pin_idx].plot(grouped_innovus['lness'], grouped_innovus['hpwl'], 'o--', label='Innovus HPWL', linewidth=2, markersize=6)
            axes[ar_idx, pin_idx].plot(grouped_innovus['lness'], grouped_innovus['rsmt'], 's--', label='Innovus RSMT', linewidth=2, markersize=6)
            # axes[ar_idx, pin_idx].plot(grouped_innovus['lness'], grouped_innovus['grwl'], '^--', label='Innovus GRWL', linewidth=2, markersize=6)
            
            # 设置每个子图的标题和标签
            axes[ar_idx, pin_idx].set_xlabel('R(P)/B(P)')
            axes[ar_idx, pin_idx].set_ylabel('Mean Wirelength')
            axes[ar_idx, pin_idx].set_title(f'WL vs R/B for p = {pin_num} (AR = {ar})')
            
            # 设置坐标轴范围和刻度
            axes[ar_idx, pin_idx].set_xlim(0.2, 0.8)
            axes[ar_idx, pin_idx].set_xticks(target_lness)
            
            # 添加网格
            axes[ar_idx, pin_idx].grid(True, alpha=0.3)
            
            # 添加图例
            axes[ar_idx, pin_idx].legend()
            
            # 设置 y 轴为科学计数法
            axes[ar_idx, pin_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()

def plot_wirelength_feature_distribution_same_pin(df_ieda, df_innovus, output_fig):
    """
    绘制 iEDA 和 Innovus 的 WL 特性分布对比图，确保相同条件下样本数相等
    参数:
        df_ieda: 包含 iEDA 数据的 DataFrame
        df_innovus: 包含 Innovus 数据的 DataFrame
        output_fig: 输出图像文件路径
    """
    # 定义目标 lness 值和其他参数
    target_lness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tolerance = 0.045
    pin_nums = [4, 5, 7]
    aspect_ratios = [1, 2, 4]

    # 创建三行三列的子图
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    plt.style.use('seaborn')
    plt.gcf().set_facecolor('white')

    def get_balanced_data(df_ieda_subset, df_innovus_subset, target, tolerance):
        """
        为特定条件下的数据集平衡样本数
        """
        # 获取指定 lness 范围内的数据
        mask_ieda = (df_ieda_subset['lness'] >= target - tolerance) & (df_ieda_subset['lness'] < target + tolerance)
        mask_innovus = (df_innovus_subset['lness'] >= target - tolerance) & (df_innovus_subset['lness'] < target + tolerance)
        
        subset_ieda = df_ieda_subset[mask_ieda]
        subset_innovus = df_innovus_subset[mask_innovus]
        
        # 确定最小样本数
        min_samples = min(len(subset_ieda), len(subset_innovus))
        
        if min_samples == 0:  # 如果任一数据集为空，返回空DataFrame
            return pd.DataFrame(), pd.DataFrame()
        
        # 随机采样到相同样本数
        if len(subset_ieda) > min_samples:
            subset_ieda = subset_ieda.sample(n=min_samples, random_state=42)
        if len(subset_innovus) > min_samples:
            subset_innovus = subset_innovus.sample(n=min_samples, random_state=42)
        
        return subset_ieda, subset_innovus

    # 对每个 aspect_ratio 和 pin_num 组合绘制子图
    for ar_idx, ar in enumerate(aspect_ratios):
        for pin_idx, pin_num in enumerate(pin_nums):
            # 筛选特定 pin_num 和 aspect_ratio 的数据
            df_pin_ieda = df_ieda[(df_ieda['pin_num'] == pin_num) & (df_ieda['aspect_ratio'] == ar)]
            df_pin_innovus = df_innovus[(df_innovus['pin_num'] == pin_num) & (df_innovus['aspect_ratio'] == ar)]
            
            # 创建空的列表来存储结果
            results_ieda = []
            results_innovus = []
            sample_sizes = []  # 存储每个点的样本数
            
            # 对每个目标 lness 值进行数据筛选和汇总
            for target in target_lness:
                # 获取平衡后的数据
                balanced_ieda, balanced_innovus = get_balanced_data(df_pin_ieda, df_pin_innovus, target, tolerance)
                
                if not balanced_ieda.empty and not balanced_innovus.empty:
                    # 计算平均值
                    sums_ieda = balanced_ieda[['hpwl', 'rsmt', 'grwl']].mean()
                    sums_innovus = balanced_innovus[['hpwl', 'rsmt', 'grwl']].mean()
                    
                    results_ieda.append({
                        'lness': target,
                        'hpwl': sums_ieda['hpwl'],
                        'rsmt': sums_ieda['rsmt'],
                        'grwl': sums_ieda['grwl']
                    })
                    
                    results_innovus.append({
                        'lness': target,
                        'hpwl': sums_innovus['hpwl'],
                        'rsmt': sums_innovus['rsmt'],
                        'grwl': sums_innovus['grwl']
                    })
                    
                    sample_sizes.append(len(balanced_ieda))  # 记录样本数
                else:
                    # 如果没有数据，添加None以保持索引对齐
                    results_ieda.append({
                        'lness': target,
                        'hpwl': None,
                        'rsmt': None,
                        'grwl': None
                    })
                    results_innovus.append({
                        'lness': target,
                        'hpwl': None,
                        'rsmt': None,
                        'grwl': None
                    })
                    sample_sizes.append(0)
            
            # 转换结果为 DataFrame
            grouped_ieda = pd.DataFrame(results_ieda)
            grouped_innovus = pd.DataFrame(results_innovus)
            
            # 设置子图的背景色
            axes[ar_idx, pin_idx].set_facecolor('#F0F0F8')
            
            # 在对应的子图上绘制曲线
            valid_mask = grouped_ieda['hpwl'].notna()  # 只绘制有效数据点
            
            if valid_mask.any():  # 如果有有效数据点
                # 绘制 iEDA 的曲线
                axes[ar_idx, pin_idx].plot(grouped_ieda.loc[valid_mask, 'lness'], 
                                         grouped_ieda.loc[valid_mask, 'hpwl'], 
                                         'o-', label='iEDA HPWL', linewidth=2, markersize=6)
                axes[ar_idx, pin_idx].plot(grouped_ieda.loc[valid_mask, 'lness'], 
                                         grouped_ieda.loc[valid_mask, 'rsmt'], 
                                         's-', label='iEDA RSMT', linewidth=2, markersize=6)
                
                # 绘制 Innovus 的曲线
                axes[ar_idx, pin_idx].plot(grouped_innovus.loc[valid_mask, 'lness'], 
                                         grouped_innovus.loc[valid_mask, 'hpwl'], 
                                         'o--', label='Innovus HPWL', linewidth=2, markersize=6)
                axes[ar_idx, pin_idx].plot(grouped_innovus.loc[valid_mask, 'lness'], 
                                         grouped_innovus.loc[valid_mask, 'rsmt'], 
                                         's--', label='Innovus RSMT', linewidth=2, markersize=6)
                
                # 添加样本数标注
                for i, (lness, n) in enumerate(zip(grouped_ieda.loc[valid_mask, 'lness'], 
                                                [sample_sizes[j] for j in range(len(sample_sizes)) if valid_mask.iloc[j]])):
                    # 为 iEDA HPWL 添加标注
                    axes[ar_idx, pin_idx].annotate(f'n={n}', 
                                                xy=(lness, grouped_ieda.loc[valid_mask, 'hpwl'].iloc[i]),
                                                xytext=(0, -15),
                                                textcoords='offset points',
                                                ha='center',
                                                fontsize=8)
                    
                    # 为 Innovus HPWL 添加标注
                    axes[ar_idx, pin_idx].annotate(f'n={n}', 
                                                xy=(lness, grouped_innovus.loc[valid_mask, 'hpwl'].iloc[i]),
                                                xytext=(0, 15),  # 向下偏移以避免重叠
                                                textcoords='offset points',
                                                ha='center',
                                                fontsize=8)
                        
            # 设置每个子图的标题和标签
            axes[ar_idx, pin_idx].set_xlabel('R(P)/B(P)')
            axes[ar_idx, pin_idx].set_ylabel('Mean Wirelength')
            axes[ar_idx, pin_idx].set_title(f'WL vs R/B for p = {pin_num} (AR = {ar})')
            
            # 设置坐标轴范围和刻度
            axes[ar_idx, pin_idx].set_xlim(0.2, 0.8)
            axes[ar_idx, pin_idx].set_xticks(target_lness)
            
            axes[ar_idx, pin_idx].grid(True, alpha=0.3)
            axes[ar_idx, pin_idx].legend()
            axes[ar_idx, pin_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 添加总标题说明样本平衡
    plt.suptitle('Wirelength Feature Distribution (Balanced samples for each condition)', 
                 fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()

# 主函数
def __main__():
    # 读取数据
    df_ieda = pd.read_csv('csv/iEDA_clean_nets.csv')
    df_innovus = pd.read_csv('csv/innovus_clean_nets.csv')

    # 绘制概率分布图
    plot_distributions_same_pin(df_ieda, df_innovus, 'fig/iEDA_vs_Innovus_Lness_distribution.png')
    
    # 进行统计检验
    results_df = perform_statistical_tests(df_ieda, df_innovus, 'csv/statistical_test_results.csv')
    print("Statistical Test Results:")
    print(results_df)

    # 绘制均值趋势图
    plot_mean_trend_with_bootstrap_ci_same_pin(df_ieda,df_innovus, 'fig/mean_trend_with_bootstrap_ci.png')
    
    # 绘制线长特征分布图
    plot_wirelength_feature_distribution_same_pin(df_ieda, df_innovus, 'fig/wirelength_feature_distribution.png')


# 调用主函数
if __name__ == "__main__":
    __main__()
