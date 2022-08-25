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

# Trainer
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

def main(args):
    dict_args = vars(args)
    # logger = TensorBoardLogger()
    trainer = pl.Trainer.from_argparse_args(args)

    if args.model_name == 'netsp':
        args.model = NETSP(**dict_args)

    model = LitNETSP(**dict_args)
    dm = LitTSPDataModule(**dict_args)


    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    # Program specific arguments
    parser.add_argument('--gpu_id', type=str, default='7')
    parser.add_argument('--model_name', type=str, default='netsp')
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