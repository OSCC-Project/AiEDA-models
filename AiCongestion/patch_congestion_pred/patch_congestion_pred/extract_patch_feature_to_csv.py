import os
import json
import csv
import numpy as np 
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  

def process_single_file(filepath):
    """处理单个JSON文件，返回提取的数据行"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取字段（根据实际JSON结构调整）
        row = [
            data.get('id', 'N/A'),
            data.get('patch_id_row', np.nan),
            data.get('patch_id_col', np.nan),
            data.get('area', np.nan),
            data.get('cell_density', np.nan),
            data.get('pin_density', np.nan),
            data.get('net_density', np.nan),
            data.get('macro_margin', np.nan),
            data.get('RUDY_congestion', np.nan),
            data.get('EGR_congestion', np.nan),
            data.get('timing', np.nan),
            data.get('power', np.nan),
            data.get('IR_drop', np.nan)
        ]
        return row
    except Exception as e:
        print(f"\nError processing {os.path.basename(filepath)}: {str(e)}")
        return None

def process_all_files(input_dirs, output_dir, max_workers=8):
    """多线程处理所有文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有JSON文件路径
    all_files = []
    for input_dir in input_dirs:
        json_files = [
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.endswith('.json')
        ]
        all_files.extend(json_files)
    
    # 多线程处理
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用tqdm显示进度条（可选）
        futures = [executor.submit(process_single_file, filepath) for filepath in all_files]
        for future in tqdm(futures, desc="Processing files", unit="file"):
            results.append(future.result())
    
    # 按项目分组写入CSV
    project_data = {}
    for filepath, row in zip(all_files, results):
        if row is None:
            continue
        project_name = filepath.split('/')[4]  # 根据路径结构调整
        output_csv = os.path.join(output_dir, f"{project_name}_patch_features_new.csv")
        project_data.setdefault(output_csv, []).append(row)
    
    # 写入CSV文件
    for csv_path, rows in project_data.items():
        header = [
            'id', 'patch_id_row', 'patch_id_col', 
            'area', 'cell_density', 'pin_density', 'net_density', 'macro_margin',
            'RUDY_congestion', 'EGR_congestion', 'timing', 'power', 'IR_drop'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

if __name__ == "__main__":
    # 输入目录列表（根据实际路径修改）
    '''input_directorys = [
        "/data2/project_share/dataset_baseline/aes/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/aes_core/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_i2c/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_ps2/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_pwm/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_rng/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_timer/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_uart/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_wdg/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/blabla/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/BM64/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/gcd/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/jpeg_encoder/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/picorv32/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/PPU/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s13207/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_archinfo/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/apb4_clint/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s1238/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s1488/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s15850/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s35932/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s38417/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s38584/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s44/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s5378/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s713/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/s9234/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/salsa20/workspace/output/innovus/feature/large_model/patchs",
        "/data2/project_share/dataset_baseline/eth_top/workspace/output/innovus/feature/large_model/patchs",
    ]'''
    input_directorys = [
        "/data2/project_share/dataset_baseline/aes/workspace/output/innovus/feature/large_model/patchs",
    ]
    
    # 输出目录
    output_directory = "/home/zhanghongda/AI4EDA/AiCongestion/patch_congestion_pred/model_new-test1"
    
    # 启动处理（线程数建议为CPU核心数的2-4倍）
    process_all_files(
        input_dirs=input_directorys,
        output_dir=output_directory,
        max_workers=4  # 根据机器性能调整
    )
