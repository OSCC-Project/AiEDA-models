import glob
import os
import pandas as pd
import numpy as np


def find_csv_files(base_directory, pattern, subdirectory_filter=None, allowed_directories=None):
    """
    查找匹配的 CSV 文件，并根据需要过滤子目录和限制搜索范围。

    :param base_directory: 根目录
    :param pattern: 文件名匹配模式
    :param subdirectory_filter: 子目录过滤条件 (如 "/output/iEDA" 或 "/output/innovus")
    :param allowed_directories: 限制搜索的目录列表
    :return: 符合条件的文件列表
    """
    all_files = []

    # 如果指定了 allowed_directories，则只在这些目录中搜索
    if allowed_directories:
        for allowed_dir in allowed_directories:
            search_pattern = os.path.join(allowed_dir, '**', pattern)
            files = glob.glob(search_pattern, recursive=True)
            all_files.extend(files)
    else:
        # 否则搜索整个 base_directory
        search_pattern = os.path.join(base_directory, '**', pattern)
        all_files = glob.glob(search_pattern, recursive=True)

    # 如果指定了子目录过滤条件，进行过滤
    if subdirectory_filter:
        all_files = [f for f in all_files if subdirectory_filter in f]

    return all_files



def output_combine_data(csv_files, output_file):
    """
    组合 CSV 文件并输出到一个文件中。
    :param csv_files: CSV 文件路径列表
    :param output_file: 输出文件名
    """
    import pandas as pd

    combined_data = pd.DataFrame()
    for file in csv_files:
        data = pd.read_csv(file)
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    combined_data.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")


def combined_origin_data(subdirectory_filter, output_file):
    base_directory = '/data/project_share/dataset_baseline/'
    pattern = '*_place_eval.csv'

    # 限制搜索的目录列表
    allowed_directories = [
        "/data/project_share/dataset_baseline/aes",
        "/data/project_share/dataset_baseline/aes_core",
        "/data/project_share/dataset_baseline/apb4_archinfo",
        "/data/project_share/dataset_baseline/apb4_clint",
        "/data/project_share/dataset_baseline/apb4_i2c",
        "/data/project_share/dataset_baseline/apb4_ps2",
        "/data/project_share/dataset_baseline/apb4_pwm",
        "/data/project_share/dataset_baseline/apb4_rng",
        "/data/project_share/dataset_baseline/apb4_timer",
        "/data/project_share/dataset_baseline/apb4_uart",
        "/data/project_share/dataset_baseline/apb4_wdg",
        "/data/project_share/dataset_baseline/blabla",
        "/data/project_share/dataset_baseline/BM64",
        "/data/project_share/dataset_baseline/eth_top",
        "/data/project_share/dataset_baseline/gcd",
        "/data/project_share/dataset_baseline/jpeg_encoder",
        "/data/project_share/dataset_baseline/picorv32",
        "/data/project_share/dataset_baseline/PPU",
        "/data/project_share/dataset_baseline/s44",
        "/data/project_share/dataset_baseline/s713",
        "/data/project_share/dataset_baseline/s1238",
        "/data/project_share/dataset_baseline/s1488",
        "/data/project_share/dataset_baseline/s5378",
        "/data/project_share/dataset_baseline/s9234",
        "/data/project_share/dataset_baseline/s13207",
        "/data/project_share/dataset_baseline/s15850",
        "/data/project_share/dataset_baseline/s35932",
        "/data/project_share/dataset_baseline/s38417",
        "/data/project_share/dataset_baseline/s38584",
        "/data/project_share/dataset_baseline/salsa20"
    ]

    # 找到所有匹配的 CSV 文件
    csv_files = find_csv_files(base_directory, pattern, subdirectory_filter, allowed_directories)

    if not csv_files:
        print("No files found matching the pattern.")
    else:
        print("Found files:", csv_files)
        print("Number of files found:", len(csv_files))

    # 组合并输出数据
    output_combine_data(csv_files, output_file)  # 可改为 'innovus_origin_nets.csv' 

def clean_origin_data(csv_file, output_file):
    # 读取原始线网数据集
    origin_nets = pd.read_csv(csv_file)

    # 数据清洗
    origin_nets['net_type'] = origin_nets['net_type'].astype(str)
    origin_nets = origin_nets[origin_nets['pin_num'] <= 31]
    origin_nets = origin_nets[origin_nets['aspect_ratio'] <= 5]
    origin_nets = origin_nets[(origin_nets['bbox_height'] > 0) & (origin_nets['bbox_width'] > 0) & (origin_nets['grwl'] > 0)]
    origin_nets.to_csv(output_file, index=False)

def show_data_info(csv_file):
    clean_nets = pd.read_csv(csv_file)
    # print(clean_nets.head())
    print(clean_nets.describe().T)
    print(clean_nets.info())


def main():
    # 构建原始线网数据集
    combined_origin_data('/output/iEDA',"csv/iEDA_origin_nets.csv" )
    combined_origin_data('/output/innovus',"csv/innovus_origin_nets.csv" )

    # 清洗原始线网数据集
    clean_origin_data('csv/iEDA_origin_nets.csv', "csv/iEDA_clean_nets.csv")
    clean_origin_data('csv/innovus_origin_nets.csv', "csv/innovus_clean_nets.csv")

    # 展示清洗后的数据集信息
    show_data_info('csv/iEDA_clean_nets.csv')
    show_data_info('csv/innovus_clean_nets.csv')



if __name__ == "__main__":
    main()
