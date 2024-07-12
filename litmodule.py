import torch
import torch.nn.functional as F
import lightning as L
from litdata import LitTSPDataModule
from model import NETSP

class LitNETSP(L.LightningModule):
    def __init__(self, model_name, lr, **kwargs):
        super().__init__()
        if model_name == 'netsp':
            self.model = NETSP(**kwargs)
        else:
            raise ValueError(f'Unknown model name {model_name}')
        self.lr = lr
    
    # @staticmethod
    # def add_model_arguments(parent_parser):
    #     parser = parent_parser.add_argument_group('Model')
    #     parser.add_argument('--d_hidden', type=int, default=128)
    #     parser.add_argument('--n_layer', type=int, default=256, help='number of encoder lstm layers')
    #     parser.add_argument('--seq_len', type=int, default=10)
    #     parser.add_argument('--bsz', type=int, default=100)
    #     parser.add_argument('--lr', type=float, default=1e-3)
    #     return parent_parser

    def compute_tour_length(self, tour, x): 
        """
        tour : (B, N)
        x : (B, 2, N)
        """
        bsz = x.shape[0]
        nb_nodes = x.shape[2]
        arange_vec = torch.arange(bsz, device=x.device)
        first_cities = x[arange_vec, :, tour[:,0]] # size(first_cities)=(bsz,2)
        previous_cities = first_cities
        L = torch.zeros(bsz, device=x.device)
        with torch.no_grad():
            for i in range(1,nb_nodes):
                current_cities = x[arange_vec, :, tour[:,i]] 
                L += torch.sum( (current_cities - previous_cities)**2 , dim=1 )**0.5 # dist(current, previous node) 
                previous_cities = current_cities
            L += torch.sum( (current_cities - first_cities)**2 , dim=1 )**0.5 # dist(last, first node)  
        return L
    
    def training_step(self, batch, batch_idx):
        x, target = batch  # (B, 2, L), (B, L)
        tour, heatmap = self.model(x, target)  # (B, L), (B, L, L)
        loss = F.nll_loss(heatmap, target)
        tour_len = self.compute_tour_length(tour, x)
        self.log('tour_len', tour_len.mean(), prog_bar=True, on_step=True)  # TODO:
        return {'loss': loss, 'tour_len': tour_len}
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        tour, heatmap = self.model(x, target)
        val_loss = F.nll_loss(heatmap, target)
        tour_len = self.compute_tour_length(tour, x)
        # self.log_dict({'val_loss': val_loss.mean().item(), 'val_tour_len': tour_len.mean().item()})
        self.log_dict({'val_loss': val_loss.mean(), 'val_tour_len': tour_len.mean()})
    
    def test_step(self, batch, batch_idx):
        x, target = batch
        tour, heatmap = self.model(x, target)
        test_loss = F.nll_loss(heatmap, target)
        tour_len = self.compute_tour_length(tour, x)
        self.log_dict({'test_loss': test_loss.mean().item(), 'test_tour_len': tour_len.mean().item()})
    
    def predict_step(self, batch):
        x, target = batch
        tour, heatmap = self.model(x, target)
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
    trainer = L.Trainer(max_epochs=-1, accelerator='cpu', devices=1)
    trainer.fit(model, dm)
