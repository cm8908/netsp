from litmodule import LitNETSP
from litdata import LitTSPDataModule
from lightning.pytorch.cli import LightningCLI
import lightning as L

if __name__ == '__main__':
    cli = LightningCLI(
        LitNETSP,
        LitTSPDataModule,
        trainer_class=L.Trainer,
        # save_config_kwargs=True,
        # run=False  # prevents the trainer from running automatically
    )

    # print(torch.cuda.is_available())
    # cli.trainer.fit(cli.model, cli.datamoduldde)
