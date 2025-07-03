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
    parser.add_argument("--des_class",type=str, default='EPFL', choices=['core','EPFL'], help="Select task category calssify or QoR")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training and testing.")
    parser.add_argument("--support_set", type=int, default= 1, help="Size of the support set.")
    parser.add_argument("--num_epochs", type=int, default = 300, help="Number of training epochs.")
    parser.add_argument("--loss", type=float, default=0.333, help="Learning rate for training.")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="path to load checkpoints.")
    parser.add_argument("--results_dir", type=str, default="../results", help="File to save results.")

    return parser.parse_args()

# 定义主函数
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.join(args.root_dir, args.des_class)
    if (os.path.exists(root_dir)==False):
            os.makedirs(root_dir)
    results_dir = os.path.join(args.results_dir, args.target)

    model = CrossLO(batch_size=args.batch_size)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_dir))
    target_ = 'test' 
    MAPES = []
    DESIGNS = []
    rhos = []
    p_values = []
    for design in design1:
        des_dir = os.path.join(root_dir, design)
        
        data = DSEDataset(dataset=des_dir, design=design, target=target_)
  
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        all_preds = []
        syns = []
        for data in tqdm(loader, desc="Evaluating design {}".format(design)):
            data = data.to(device)
            
            # 调用模型的 dse 方法进行推理
            with torch.no_grad():
                pred = model(data)
            all_preds.append(pred)

        all_preds = torch.cat(all_preds)
        
        if target_ == 'test':
            test_csv = os.path.join(des_dir, 'eval.csv')
            new_csv = os.path.join(des_dir, f'mlp{design}{target_}.csv')
            design_dir = os.path.join(des_dir,f'des_{args.target}.csv')
            mean_, std_ = load_mean_std(design_dir)
            all_preds_np = all_preds.cpu().numpy()
            df = pd.read_csv(test_csv)
            df[f'{args.target}pred'] = all_preds_np*std_ + mean_
            df.to_csv(new_csv, index=False)

            rho, p_value = spearmanr(df[f'{args.target}pred'], df[f'{args.target}'])
            mape = mean_absolute_percentage_error(df[f'{args.target}'], df[f'{args.target}pred'])

            print(f'{design}: MAPE {mape}, rho {rho}, p_value {p_value}')
            DESIGNS.append(design)
            MAPES.append(mape)
            rhos.append(rho)
            p_values.append(p_value)
            plt.figure(figsize=(10, 6))
            plt.scatter(df[f'{args.target}'], df[f'{args.target}pred'], alpha=0.5)

            plt.title(f'{args.target} MAPE: {mape*100:.2f}%', fontsize=25, weight='bold')
            plt.xlabel('Actual', fontsize=25, weight='bold')
            plt.ylabel('Predicted', fontsize=25, weight='bold')

            save_dir = os.path.join(des_dir, f'mlp{args.target}test.png')
            plt.savefig(save_dir)
        elif target_ == 'train':
            train_csv = os.path.join(des_dir, 'character.csv')
            new_csv = os.path.join(des_dir, f'unseen{design}_{target_}.csv')
            design_dir = os.path.join(des_dir,f'des_{args.target}.csv')
            mean_, std_ = load_mean_std(design_dir)

            all_preds_np = all_preds.cpu().numpy()
            df = pd.read_csv(train_csv)
            df[f'{args.target}pred'] = all_preds_np*std_ + mean_

            df.to_csv(new_csv, index=False)

            rho, p_value = spearmanr(df[f'{args.target}pred'], df[f'{args.target}'])
            mape = mean_absolute_percentage_error(df[f'{args.target}'], df[f'{args.target}pred'])

            print(f'{design}: MAPE {mape}, rho {rho}, p_value {p_value}')
            DESIGNS.append(design)
            MAPES.append(mape)
            rhos.append(rho)
            p_values.append(p_value)
            plt.figure(figsize=(10, 6))
            plt.scatter(df[f'{args.target}'], df[f'{args.target}pred'], alpha=0.5)

            plt.title(f'{args.target} MAPE: {mape*100:.2f}%', fontsize=25, weight='bold')
            plt.xlabel('Actual', fontsize=25, weight='bold')
            plt.ylabel('Predicted', fontsize=25, weight='bold')

            save_dir = os.path.join(des_dir, f'{args.target}{target_}.png')
            plt.savefig(save_dir)
            plt.close()
    df = pd.DataFrame({'DESIGNS': DESIGNS, 'MAPES': MAPES, 'rhos': rhos, 'p_values': p_values})
    print(f'MAPE mean: {np.mean(MAPES)}, rho mean: {np.mean(rhos)}, p_values mean: {np.mean(p_values)}')
    df_dir = os.path.join(root_dir, f'CrossLO_nuseen_IC_{args.target}{target_}_{args.loss}summary.csv')
    df.to_csv(df_dir, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    