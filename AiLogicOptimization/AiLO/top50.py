import os
import re
import random
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import argparse
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mean_absolute_percentage_error
import torch
from torch_geometric.loader import DataLoader

from model.gnntr_train import CrossLO
from dataset.dataset import DSEDataset
from dataset.utils import *

OptDict = {
    1: "refactor",
    2: "refactor -z",
    3: "refactor -l",
    4: "refactor -l -z",
    5: "rewrite",
    6: "rewrite -z",
    7: "rewrite -l",
    8: "rewrite -l -z" ,
    9: "resub",
    10: "resub -z",
    11: "resub -l",
    12: "resub -l -z",
    13: "balance"
}

def load_mean_std(csv_dir):
    df = pd.read_csv(csv_dir)
    mean_ ,std_ = df['mean'].iloc[0], df['std'].iloc[0]
    return mean_, std_

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN model and save checkpoints.")

    parser.add_argument("--root_dir", type=str, required=False, default='', help="Root directory for dataset.")
    parser.add_argument("--target",type=str, default='delay', choices=['area','delay'], help="Select task category calssify or QoR")
    parser.add_argument("--des_class",type=str, default='EPFL', choices=['core','comb'], help="Select task category calssify or QoR")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training and testing.")
    parser.add_argument("--gnn", type=str,default='gin', choices=['gin','graphsage', 'gcn'], help="Type of GNN model.")
    parser.add_argument("--support_set", type=int, default= 1, help="Size of the support set.")
    parser.add_argument("--num_epochs", type=int, default = 300, help="Number of training epochs.")
    parser.add_argument("--loss", type=float, default=0.333, help="Learning rate for training.")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory to save checkpoints.")
    parser.add_argument("--results_dir", type=str, default="../results", help="File to save results.")

    return parser.parse_args()

# 定义主函数
def main(args):
    # 使用参数初始化模型和设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.join(args.root_dir, args.des_class)
    if (os.path.exists(root_dir)==False):
            os.makedirs(root_dir)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.target)
    results_dir = os.path.join(args.results_dir, args.target)

    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir)

    delay_model = CrossLO(batch_size=args.batch_size)
    area_model = CrossLO(batch_size=args.batch_size)
    delay_model.to(device)
    area_model.to(device)
    delay_model.load_state_dict(torch.load(''))
    area_model.load_state_dict(torch.load(''))
    target_ = 'test' 
    hit_rs = []
    DESIGNS = []
    rhos = []
    p_values = []
    for design in design1:
        des_dir = os.path.join(root_dir, design)
        
        data = DSEDataset(dataset=des_dir, design=design, target=target_)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        all_delay_preds = []
        all_area_preds = []
        syns = []
        for data in tqdm(loader, desc="Evaluating design {}".format(design)):
            data = data.to(device)
            with torch.no_grad():
                delay_pred = delay_model(data)
                area_pred = area_model(data)
            all_delay_preds.append(delay_pred)
            all_area_preds.append(area_pred)


        all_delay_preds = torch.cat(all_delay_preds)
        all_area_preds = torch.cat(all_area_preds)

        if target_ == 'test':
            test_csv = os.path.join(des_dir, 'eval.csv')
            new_csv = os.path.join(des_dir, f'mlp{design}{target_}.csv')
            delay_design_dir = os.path.join(des_dir,'des_delay.csv')
            area_design_dir = os.path.join(des_dir,'des_area.csv')
            delay_mean_, delay_std_ = load_mean_std(delay_design_dir)
            area_mean_, area_std_ = load_mean_std(area_design_dir)

            all_delay_preds_np = all_delay_preds.cpu().numpy()
            all_area_preds_np = all_area_preds.cpu().numpy()
            df = pd.read_csv(test_csv)

            df['delay'] = (df['delay'] - delay_mean_)/ delay_std_
            df['area'] = (df['area'] - area_mean_) / area_std_
            df['QoR'] = df['delay'] + df['area']
            df['delay_pred'] = all_delay_preds_np
            df['area_pred'] = all_area_preds_np
            df['QoR_pred'] = df['delay_pred'] + df['area_pred']
            df_sorted_pred = df.sort_values(by='QoR_pred', ascending=False)
            top_50_pred = df_sorted_pred.head(50)
            df_sorted_true = df.sort_values(by='QoR', ascending=False)
            top_50_true = df_sorted_true.head(50)
            hit_count = len(set(top_50_pred.index) & set(top_50_true.index))
            hit_rate = (hit_count / 50) * 100

            print(f'命中率：{hit_rate:.2f}%')
            hit_rs.append(hit_rate)
            DESIGNS.append(design)
    hit = pd.DataFrame({'DESIGNS':DESIGNS, 'hit':hit_rs})
    hit_dir = os.path.join(root_dir, f'CrossLO_hitQoR{target_}.csv')
    hit.to_csv(hit_dir, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    