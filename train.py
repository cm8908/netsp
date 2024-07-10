import os
import pytorch_lightning as pl
from litmodule import LitNETSP
from litdata import LitTSPDataModule
from pytorch_lightning.cli import LightningCLI

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--gpu_id', type=str, default='0')

    def before_instantiate_classes(self):
        args = self.config.fit
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if __name__ == '__main__':
    cli = CustomLightningCLI(
        LitNETSP,
        LitTSPDataModule,
        trainer_class=pl.Trainer,
        # save_config_kwargs=True,
        # run=False  # prevents the trainer from running automatically
    )

    # print(torch.cuda.is_available())
    # cli.trainer.fit(cli.model, cli.datamoduldde)
