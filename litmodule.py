import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from litdata import LitTSPDataModule
from model import NETSP
from torch.utils.data import Dataset


class LitNETSP(pl.LightningModule):
    def __init__(self, model: NETSP, lr: float=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
    
    @staticmethod
    def add_model_arguments(parent_parser):
        parser = parent_parser.add_argument_group('Model')
        parser.add_argument('--d_hidden', type=int, default=128)
        parser.add_argument('--seq_len', type=int, default=10)
        parser.add_argument('--bsz', type=int, default=100)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parent_parser

    def compute_tour_length(self, tour, x):
        """
        tour : (B, N)
        x : (B, 2, N)
        """
        tour = tour.cpu().numpy()
        x = x.cpu().numpy()
        bsz = x.shape[0]
        nb_nodes = x.shape[2]
        toB = np.arange(bsz)
        first_cities = x[toB, :,  tour[:,0]] # size(first_cities)=(bsz,2)
        previous_cities = first_cities
        L = np.zeros(bsz)
        for i in range(1,nb_nodes):
            current_cities = x[toB, :, tour[:,i]] 
            L += np.sum( (current_cities - previous_cities)**2 , axis=1 )**0.5 # dist(current, previous node) 
            previous_cities = current_cities
        L += np.sum( (current_cities - first_cities)**2 , axis=1 )**0.5 # dist(last, first node)  
        return L
    
    def training_step(self, batch, batch_idx):
        x, target = batch  # (B, 2, L), (B, L)
        tour, heatmap = self.model(x)  # (B, L), (B, L, L)
        loss = F.nll_loss(heatmap, target)
        tour_len = self.compute_tour_length(tour, x)
        return {'loss': loss, 'tour_len': tour_len}
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        tour, heatmap = self.model(x)
        val_loss = F.nll_loss(heatmap, target)
        tour_len = self.compute_tour_length(tour, x)
        self.log_dict({'val_loss': val_loss.mean().item(), 'val_tour_len': tour_len.mean().item()})
    
    def test_step(self, batch, batch_idx):
        x, target = batch
        tour, heatmap = self.model(x)
        test_loss = F.nll_loss(heatmap, target)
        tour_len = self.compute_tour_length(tour, x)
        self.log_dict({'test_loss': test_loss.mean().item(), 'test_tour_len': tour_len.mean().item()})
    
    def predict_step(self, batch):
        x, target = batch
        tour, heatmap = self.model(x)
        return tour

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # lr_scheduler = None
        # return [optimizer], [lr_scheduler]
        return optimizer
    

if __name__ == '__main__':
    n = 10
    d_h = 128
    bsz = 512
    batch_first = True
    model = LitNETSP(NETSP(d_h, bsz, n, batch_first))
    dm = LitTSPDataModule(n, bsz, num_workers=40)
    trainer = pl.Trainer(max_epochs=-1, accelerator='cpu', devices=1)
    trainer.fit(model, dm)