import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
import pandas as pd
import numpy as np
import os
import time
from aifp.database.pretrain_dataset import create_train_and_valid_set
from aifp.operation.data_io.adj_reader import read_adj
from torch.utils.data import DataLoader
from aifp import setting

pretrain_config = {
    'optimizer': 'Adam',
    'lr': 5e-4,
    'epoch': 100,
    'minibatch_size': 128,
}

def set_random_seeds():
    torch.manual_seed(setting.rl_config['torch_seed'])
    np.random.seed(setting.rl_config['numpy_seed'])

def set_logging_dir():
    log_dir = os.environ["AIFP_PATH"] + setting.log['log_dir'] + '/run{}/'.format(setting.log['run_num'])
    model_dir = os.environ["AIFP_PATH"] + setting.log['model_dir'] + '/run{}/'.format(setting.log['run_num'])
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    logging.basicConfig(filename=log_dir + 'logging.log', level=logging.INFO)
    logging.root.name = 'aifp:pretrainer'

class PreTrainer:
    def __init__(self, case_select:str, model:nn.Module):
        self._device = torch.device('cuda' if (setting.pretrain['device'] == 'cuda' and torch.cuda.is_available()) else 'cpu')
        self._model = model.to(self._device)
        self._model._device = self._device # set model's deivce
        self._optimizer = self._get_optimizer()
        self._loss_fn = nn.MSELoss()
        self._train_set, self._valid_set = create_train_and_valid_set()
        self._train_dataloader = DataLoader(self._train_set, setting.pretrain['batch_size'], shuffle=True, num_workers=setting.pretrain['dataloader_worker'])
        self._valid_dataloader = DataLoader(self._valid_set, setting.pretrain['batch_size'], shuffle=False, num_workers=setting.pretrain['dataloader_worker'])
        log_dir = os.environ['AIFP_PATH'] + setting.log['log_dir'] + 'run{}'.format(setting.log['run_num'])
        set_random_seeds()
        set_logging_dir()
        self._writer = SummaryWriter(log_dir=log_dir)

    def _get_optimizer(self):
        optimizer_name = setting.pretrain['optimizer']
        if optimizer_name == 'Adam':
            return optim.Adam(params=self._model.parameters(), lr=setting.pretrain['lr'])
        else:
            raise NotImplementedError
    
    def save_model(self, save_path:str):
        torch.save(self._model.state_dict(), save_path)
        print('===== pretrained model saved to {} ======'.format(save_path))
    
    def load_model(self, load_path:str):
        self._model.load_state_dict(torch.load(load_path))
        print('===== pretrained model loaded from {} ====='.format(load_path))

    def train(self):
        for epoch in range(setting.pretrain['epoch']):
            train_loss = self._train_loop()
            valid_loss = self._valid_loop()
            self._writer.add_scalars(main_tag='loss', tag_scalar_dict = {'train': train_loss, 'valid': valid_loss}, global_step=epoch)
            print('epoch {}, train_loss: {:.10f}, valid_loss: {:.10f}'.format(epoch, train_loss, valid_loss))
            logging.info('epoch {}, train_loss: {:.10f}, valid_loss: {:.10f}'.format(epoch, train_loss, valid_loss))


    def _train_loop(self):
        self._model.train()
        num_batchs = len(self._train_dataloader)
        train_loss = 0.0

        for batch_idx, (features, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, labels) in enumerate(self._train_dataloader):
            features = features.to(self._device)
            macro_idx_to_place = macro_idx_to_place.to(self._device)
            sparse_adj_i = sparse_adj_i.to(self._device)
            sparse_adj_j = sparse_adj_j.to(self._device)
            sparse_adj_weight = sparse_adj_weight.to(self._device)
            labels = labels.to(self._device)

            preds = self._model.pretrain_value(features, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight).squeeze()
            loss = self._loss_fn(preds, labels)
            self._optimizer.zero_grad()
            loss.backward()

            if setting.pretrain['use_grad_clip']:
                torch.nn.utils.clip_grad_norm_(parameters=self._model.parameters(),
                    max_norm=setting.pretrain['clip_max_norm'], norm_type=setting.pretrain['clip_norm_type'])

            self._optimizer.step()
            train_loss += loss.cpu().item()

        train_loss /= num_batchs
        return train_loss

    def _valid_loop(self):
        self._model.eval()
        num_batchs = len(self._valid_dataloader)
        valid_loss = 0.0

        with torch.no_grad():
            for batch_idx, (features, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, labels) in enumerate(self._train_dataloader):
                features = features.to(self._device)
                macro_idx_to_place = macro_idx_to_place.to(self._device)
                sparse_adj_i = sparse_adj_i.to(self._device)
                sparse_adj_j = sparse_adj_j.to(self._device)
                sparse_adj_weight = sparse_adj_weight.to(self._device)
                labels = labels.to(self._device)

                preds = self._model.pretrain_value(features, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight).squeeze()
                valid_loss += self._loss_fn(preds, labels).cpu().item()

        valid_loss /= num_batchs
        return valid_loss
    



        