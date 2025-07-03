import os
import csv
import torch
from model.gnntr_train import CrossLO, train, test
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil
import argparse
import time

from dataset.dataset import QoR_Dataset
from model.utils import random_sampler
from dataset.utils import *
from torch_geometric.loader import DataLoader

def data_split(n_tasks, root, target, batch_size):
    train_sets = []
    test_sets = []
    for train_task in range(1, n_tasks+1):
        designs = eval(f'design{train_task}')
        task_data_dir = os.path.join(root, f'design{train_task}')
        dataset = QoR_Dataset(task_data_dir, root, designs, target)
        train_data= random_sampler(dataset, train=True)
        test_data= random_sampler(dataset, train=False)
        train_set = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_set = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        train_sets.append(train_set)
        test_sets.append(test_set)
    return train_sets, test_sets

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN model and save checkpoints.")
    parser.add_argument("--root_dir", type=str, required=False, default='', help="Root directory for dataset.")
    parser.add_argument("--target",type=str, default='delay', choices=['area','delay'], help="Select task category calssify or QoR")
    parser.add_argument("--des_class",type=str, default='EPFL', choices=['core','EPFL'], help="Select task category calssify or QoR")
    parser.add_argument("--batch_size", type=int, default = 32, help="Batch size for training and testing.")
    parser.add_argument("--tasks", type=int, default=1, help="Number of design tasks.")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate for training.")
    parser.add_argument("--support_set", type=int, default= 1, help="Size of the support set.")
    parser.add_argument("--num_epochs", type=int, default = 300, help="Number of training epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/EPFL", help="Directory to save checkpoints.")
    parser.add_argument("--results_dir", type=str, default="./results/EPFL", help="File to save results.")

    return parser.parse_args()


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.join(args.root_dir, args.des_class)
    if (os.path.exists(root_dir)==False):
            os.makedirs(root_dir)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.target)
    results_dir = os.path.join(args.results_dir, args.target)

    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir)

    model = CrossLO(batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
    model.to(device)
    

    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
    
    result_csv_dir = os.path.join(results_dir,f'results_{args.gnn}_{args.support_set}_{args.target}.csv')
    if not os.path.exists(result_csv_dir):
        with open(result_csv_dir, 'w', newline='') as result_csv_file:
            csv_writer = csv.writer(result_csv_file)
            csv_writer.writerow(['Epoch', 'TrainLoss','TestLoss','Runtime','Best Loss'])
    TrainLoss = []
    TestLoss = []
    best_score = 999
    start_time = time.time()
    train_sets, test_sets = data_split(args.tasks, root_dir, args.target, args.batch_size)

    for epoch in range(1, args.num_epochs+1):
        is_best = False
        trainloss = 0
        testloss = 0
        for i in range(args.tasks):
            tasktrainloss = train(model, train_sets[i], device, optimizer)
            trainloss += tasktrainloss
        trainloss /= args.tasks
        scheduler.step(trainloss)
        TrainLoss.append(trainloss)
        print(f"Epoch: {epoch}, Train Loss: {trainloss}")
        for i in range(args.tasks):
            tasktestloss = test(model, test_sets[i], device)
            testloss += tasktestloss
        testloss /= args.tasks
        TestLoss.append(testloss)
        print(f"Epoch: {epoch}, Test Loss: {testloss}")

        if best_score > testloss:
            best_score = testloss
            is_best = True

        with open(result_csv_dir, "a", newline='') as result_csv_file:
            csv_writer = csv.writer(result_csv_file)
            runtime_str = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
            csv_writer.writerow([epoch, trainloss, testloss, runtime_str,best_score])

        if is_best and epoch > 5 and best_score < 9:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "CrossLS-epoch-{}-loss-{:.3f}.pt".format(epoch, best_score)))
        if epoch == args.num_epochs:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "CrossLS-epoch-{}-loss-{:.3f}.pt".format(epoch, best_score)))

if __name__ == "__main__":
    args = parse_args()
    main(args)
