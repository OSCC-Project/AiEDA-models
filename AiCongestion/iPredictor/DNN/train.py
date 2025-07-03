"""
Author: juanyu 291701755@qq.com
Description: train DNN model
"""

import argparse
import matplotlib.pyplot as plt
import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from DNN import DNN
from utils.model_eval import model_eval
from dataset_DNN import DNNDataset
from utils.EarlyStop import EarlyStopping


def arg_parse():
    parser = argparse.ArgumentParser(description='Train DNN model')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,help='Learning rate', dest='lr')
    parser.add_argument('--patience', '-p', type=int, default=20, help='How long to wait after last time validation loss improved')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--data', '-d', type=str, default='../data/ispd2019/all_data_0.5.npy', help='Path of dataset')
    parser.add_argument('--save-path', '-m', dest='save_dir',type=str, default='./DNN_ispd2019/', help='Save model and loss figure path')
    return parser.parse_args()

def train(n_epochs, optimizer, model, loss_fn, device, patience):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=args.save_dir + "checkpoint.pt")

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        for input, labels in train_loader:
            input = input.to(device=device).float()
            labels = labels.to(device=device).long()
            labels = torch.squeeze(labels)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(input)
            # calculate the loss
            loss = loss_fn(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            for input, labels in val_loader:
                input = input.to(device=device).float()
                labels = labels.to(device=device).long()
                labels = torch.squeeze(labels)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(input)
                # calculate the loss
                loss = loss_fn(output, labels)
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.save_dir + "checkpoint.pt"))

    return model, avg_train_losses, avg_valid_losses

if __name__ == "__main__":
    args = arg_parse()

    device = (torch.device("cuda") if torch.cuda.is_available() else ("cpu"))

    print(f"Training on device {device}")

    # get data and labels
    # Split into train / validation partitions
    DNNDataset = DNNDataset(dataset_dir=args.data)
    val_percent = args.val / 100
    n_val = int(len(DNNDataset) * val_percent)
    n_train = len(DNNDataset) - n_val
    train_set, val_set = random_split(DNNDataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    model = DNN().to(device=device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # compute weights for loss function
    num_positive = 0
    num_negative = 0
    for _, label in train_set:
        if label in [1]:
            num_positive += 1
        else:
            num_negative += 1
    weights = torch.tensor([num_negative, num_positive],
                           dtype=torch.float32,
                           device=device)
    weights = weights / weights.sum()
    weights = 1.0 / weights

    loss_fn = nn.CrossEntropyLoss()

    model, train_loss, valid_loss = train(n_epochs=args.epochs, optimizer=optimizer, model=model, loss_fn=loss_fn, device=device, patience=args.patience)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.7)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(args.save_dir + 'loss_plot.png', bbox_inches='tight')

