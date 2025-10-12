from argparse import ArgumentParser
import pytorch_lightning as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from module import IcoDenoiseNet, CustomModelCheckpoint
from data_loader import IcoMeshDataModule, IcoMeshPredictionModule
from utils.logging import SavePredictionsCallback, LogMeshCallback


def main(args):
    # create datamodule
    dm = IcoMeshDataModule.from_argparse_args(args)

    # create model
    if args.resume_from_checkpoint:
        model = IcoDenoiseNet.load_from_checkpoint(args.resume_from_checkpoint)
        if args.load_weights_only:
            args.resume_from_checkpoint = None
    else:
        dict_args = vars(args)
        model = IcoDenoiseNet(**dict_args)

    # define callbacks for checkpoints and hparams logging
    logging_callback = CustomModelCheckpoint(
        monitor="val_loss",
        hp_metrics=["val_loss", "dist", "angdiff"],
        filename="best",
        mode="min",
    )

    # log mesh callback
    log_mesh_callback = LogMeshCallback(args.log_mesh_epoch, args.log_mesh_samples)

    # define logger
    logger = TensorBoardLogger(save_dir=args.log_dir, log_graph=True, default_hp_metric=False)

    # create trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        log_every_n_steps=2,
        num_sanity_val_steps=-1,
        callbacks=[logging_callback, log_mesh_callback],
        logger=logger,
    )

    # train
    trainer.fit(model, datamodule=dm)

    if args.save_predictions:
        # create datamodule
        dm = IcoMeshPredictionModule.from_argparse_args(args)

        # define callback for saving predictions as new dataset (for cascaded networks)
        save_predictions_callback = SavePredictionsCallback(save_dir=args.save_predictions)

        # create trainer
        trainer = pl.Trainer.from_argparse_args(
            args, logger=False, checkpoint_callback=False, callbacks=[save_predictions_callback]
        )

        # test
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()

    # define run specific arguments
    parser.add_argument(
        "--save_predictions", default=None, type=str, help="store all predictions as new dataset at given path"
    )
    parser.add_argument(
        "--load_weights_only", action="store_true", help="only load the weights from the given checkpoint"
    )

    # add params from Trainer
    parser = pl.Trainer.add_argparse_args(parser)
    # add params from model
    parser = IcoDenoiseNet.add_model_specific_args(parser)
    # add params from datamodule
    parser = IcoMeshDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
