import torch
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass
class TSPDataset(Dataset):
    """
    B = total length of dataset
    self.data : total records of all n (x,y) samples (Tensor shape B x N x 2)
    self.label : concorde solutions of tsp tour index sequence (Tensor shape B x N)
    n : {10, 20, 30, 50, 100}
    Return: data, label indices  #, distance matrix, modified adjacency matrix, adjacency matrix, label tour length
    """
    def __init__(self, n, mode, root_dir='./datasets', author='PN'):
        if mode in ['train', 'val']:
            basename = f'tsp{n}_concorde.txt'
        else:
            basename = f'tsp{n}_{mode}_concorde.txt'
        filename = os.path.join(root_dir, author, basename)
        self.n = n
        self.data = []
        self.label = []
        with open(filename, 'r') as file:
            for line in tqdm(list(file)):
                if line == '\n':
                    break
                sample = line.split(' ')    
                xs = list(map(float, sample[:2*n-1:2]))
                ys = list(map(float, sample[1:2*n:2]))
                sample_data = [xs, ys]
                sample_label = list(map(int, sample[2*n+1:-1]))

                self.data.append(sample_data)
                self.label.append(sample_label)
        if len(self.data) != len(self.label):
            raise ValueError(f'length of data {len(self.data)} while label {len(self.label)}')

        # B x 2 x N, B x N
        self.data = torch.Tensor(self.data)
        self.label = torch.LongTensor(self.label)[:,:-1] - 1
        # print(self.data.shape)
        # print(self.label.shape)
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class LitTSPDataModule(LightningDataModule):
    def __init__(self, seq_len, bsz, train_ratio, num_workers, **kwargs):
        super().__init__()
        self.n = seq_len
        self.batch_size = bsz
        self.train_ratio = train_ratio
        self.num_workers = num_workers
    
   
    def setup(self, stage):
        print('Loading dataset from file')
        if stage == 'fit':
            if self.n < 50:
                self.train_set = TSPDataset(self.n, mode='train')
                self.val_set = TSPDataset(self.n, mode='val')
            else:
                dataset = TSPDataset(self.n, mode='train')
                train_size = int(len(dataset) * self.train_ratio)
                self.train_set = dataset[:train_size]
                self.val_set = dataset[train_size:]
        elif stage == 'test':
            self.test_set = TSPDataset(self.n, mode='test')
        else:
            raise ValueError('Incorrect stage specified')
        
    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=self.num_workers)

if __name__ == '__main__':
    # dataset = TSPDataset(n=10, mode='train')
    # print(len(dataset))
    # data, label = dataset[0]
    # print(data.shape, label.shape)
    n = 10
    batch_size = 500
    train_ratio = 0.8

    dm = LitTSPDataModule(n, batch_size, train_ratio)
    dm.setup(stage='fit')
    print(next(iter(dm.train_dataloader())))
