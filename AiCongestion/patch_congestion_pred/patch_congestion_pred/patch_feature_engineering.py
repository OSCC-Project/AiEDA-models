import glob
import os
import pandas as pd
import numpy as np

def combined_patch_data(input_dir, output_file):
    """
    合并所有patch特征CSV文件
    
    参数:
    input_dir -- 输入目录路径，包含多个patch特征CSV文件
    output_file -- 输出文件路径
    """
    # 确保输入目录路径正确
    if not input_dir.endswith('/'):
        input_dir = input_dir + '/'
    
    # 获取目录下所有patch特征CSV文件
    csv_files = glob.glob(input_dir + "*patch_feature*.csv")
    
    if not csv_files:
        print(f"在 {input_dir} 目录下未找到patch特征CSV文件")
        return
    
    # 创建空列表存储数据
    all_data = []
    header = None
    
    # 遍历所有CSV文件
    for i, file in enumerate(csv_files):
        try:
            # 读取CSV文件
            df = pd.read_csv(file)
            
            # 如果存在'id'列，则保留它（patch分析可能需要id）
            if 'id' in df.columns:
                df = df[['id', 'area', 'cell_density', 'pin_density', 
                         'net_density', 'macro_margin', 'RUDY_congestion', 'EGR_congestion']]
            
            # 第一个文件，保存列名
            if i == 0:
                header = df.columns.tolist()
            
            # 添加数据
            all_data.append(df)
            print(f"处理文件: {file}")
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    
    if not all_data:
        print("没有有效的数据可以合并")
        return
    
    # 合并所有数据框
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存合并后的数据
    combined_df.to_csv(output_file, index=False)
    print(f"合并完成，已保存到 {output_file}")
    print(f"合并了 {len(csv_files)} 个文件，共 {len(combined_df)} 行数据")

def clean_patch_data(csv_file, output_file):
    """
    清洗patch数据
    
    参数:
    csv_file -- 输入的合并patch CSV文件路径
    output_file -- 输出文件路径
    """
    print("\n=== 开始清洗patch数据 ===")
    
    try:
        # 读取数据
        df = pd.read_csv(csv_file)
        original_count = len(df)
        print(f"原始数据记录数: {original_count}")
        
        # 数据清洗规则
        # 1. 去除area为0或负值的记录
        df = df[df['area'] > 0]
        # 清洗掉所有特征为0但EGR不为0的数据
        df = df[~((df['cell_density'] == 0) & (df['pin_density'] == 0) & (df['net_density'] == 0) & 
                (df['macro_margin'] == 0) & (df['RUDY_congestion'] == 0) & (df['EGR_congestion'] != 0))]
        

        
        # 3. 限制congestion值在合理范围内
        #df = df[(df['RUDY_congestion'] >= -100) & (df['RUDY_congestion'] <= 100)]
        
        # 保存清洗后的数据
        cleaned_count = len(df)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"清洗后的数据记录数: {cleaned_count}")
        print(f"已移除 {original_count - cleaned_count} 条记录")
        print(f"清洗后的数据已保存至: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None

def show_patch_info(csv_file):
    """
    显示patch数据基本信息
    
    参数:
    csv_file -- CSV文件路径
    """
    print("\n=== Patch数据基本信息 ===")
    try:
        df = pd.read_csv(csv_file)
        print(df.describe().T)
        print("\n数据类型信息:")
        print(df.info())
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

def main():
    # 设置输入输出路径
    input_dir = "patch_congestion_pred/model/"
    combined_output = "patch_congestion_pred/model/combined_patches.csv"
    cleaned_output = "patch_congestion_pred/model/cleaned_patches.csv"
    
    # 1. 合并所有patch特征文件
    combined_patch_data(input_dir, combined_output)
    
    # 2. 清洗合并后的patch数据
    clean_patch_data(combined_output, cleaned_output)
    
    # 3. 显示清洗后的数据信息
    show_patch_info(cleaned_output)

if __name__ == "__main__":
    main()
