import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_pin_num_distribution_comparison(filepath1, filepath2, labels, output_path=None):
    """
    绘制两份数据的 pin_num 分布柱状图对比。
    
    参数：
    - filepath1: 第一份数据的文件路径
    - filepath2: 第二份数据的文件路径
    - labels: 两份数据的标签（如 ['iEDA', 'Innovus'])
    - output_path: 保存图片的路径
    """
    # 读取数据
    data1 = pd.read_csv(filepath1)
    data2 = pd.read_csv(filepath2)
    
    # 去除 pin_num = 3 的样本
    data1 = data1[data1['pin_num'] != 3]
    data2 = data2[data2['pin_num'] != 3]
    
    # 统计每个 pin_num 的 net 个数
    pin_num_counts1 = data1['pin_num'].value_counts()
    pin_num_counts2 = data2['pin_num'].value_counts()
    
    # 计算总数
    total_nets1 = pin_num_counts1.sum()
    total_nets2 = pin_num_counts2.sum()
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # 绘制第一份数据的柱状图
    bars1 = pin_num_counts1.sort_index().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution of Net Counts by Pin Number ({labels[0]})')
    axes[0].set_xlabel('Pin Number')
    axes[0].set_ylabel('Number of Nets')
    axes[0].set_xticks(range(len(pin_num_counts1)))
    axes[0].set_xticklabels(pin_num_counts1.sort_index().index, rotation=45)
    
    # 在每个柱子上标注百分比
    for bar in bars1.patches:
        height = bar.get_height()
        percentage = f'{100 * height / total_nets1:.1f}%'
        axes[0].annotate(percentage,
                         (bar.get_x() + bar.get_width() / 2, height),
                         ha='center', va='bottom' if height < max(pin_num_counts1) * 0.1 else 'top',
                         fontsize=10, color='black')
    
    # 绘制第二份数据的柱状图
    bars2 = pin_num_counts2.sort_index().plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title(f'Distribution of Net Counts by Pin Number ({labels[1]})')
    axes[1].set_xlabel('Pin Number')
    axes[1].set_xticks(range(len(pin_num_counts2)))
    axes[1].set_xticklabels(pin_num_counts2.sort_index().index, rotation=45)
    
    # 在每个柱子上标注百分比
    for bar in bars2.patches:
        height = bar.get_height()
        percentage = f'{100 * height / total_nets2:.1f}%'
        axes[1].annotate(percentage,
                         (bar.get_x() + bar.get_width() / 2, height),
                         ha='center', va='bottom' if height < max(pin_num_counts2) * 0.1 else 'top',
                         fontsize=10, color='black')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path)
    plt.show()



# 调用主函数
if __name__ == "__main__":
    filepath_ieda = 'csv/iEDA_clean_nets.csv'
    filepath_innovus = 'csv/innovus_clean_nets.csv'
    labels = ['iEDA', 'Innovus']
    plot_pin_num_distribution_comparison(filepath_ieda, filepath_innovus, labels, 'fig/net_pin_number_comparison.png')