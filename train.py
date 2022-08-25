import os
import pytorch_lightning as pl
from model import NETSP
from argparse import ArgumentParser
from litmodule import LitNETSP
from litdata import LitTSPDataModule
# from pytorch_lightning.loggers import TensorBoardLogger
# Specifying hyperparameters
# https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html

# Modifying progress bar
# https://forums.pytorchlightning.ai/t/how-to-modify-the-default-progress-bar/224/3

# Logger version & directory names
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#default-root-dir

def main(args):
    # logger = TensorBoardLogger()
    trainer = pl.Trainer.from_argparse_args(args)

    model = LitNETSP(
        model = NETSP(
            d_h=args.d_hidden,
            bsz=args.bsz,
            seq_len=args.seq_len
        ),
        lr=args.lr
    )
    dm = LitTSPDataModule(
        n=args.seq_len,
        bsz=args.bsz, 
        num_workers=args.num_workers, 
        train_ratio=args.train_ratio
    )

    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    # Program specific arguments
    parser.add_argument('--gpu_id', type=str, default='7')
    # parser.add_argument('--version', type=int, default=999)
    # Model specific arguments
    parser = LitNETSP.add_model_arguments(parser)
    # Data Module arguments
    parser = LitTSPDataModule.add_data_arguments(parser)
    # Trainer specific arguments
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    main(args)