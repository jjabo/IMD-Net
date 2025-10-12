from argparse import ArgumentParser
import pytorch_lightning as pl

from module import IcoDenoiseNet
from data_loader import IcoMeshPredictionModule
from utils.logging import SavePredictionsCallback, SaveMetricsCallback


def main(args):
    # create datamodule
    dm = IcoMeshPredictionModule.from_argparse_args(args)

    # create model
    model = IcoDenoiseNet.load_from_checkpoint(args.resume_from_checkpoint)

    callbacks = []
    if args.save_predictions is not None:
        callbacks.append(SavePredictionsCallback(save_dir=args.save_predictions))
    if args.save_metrics is not None:
        callbacks.append(SaveMetricsCallback(save_file=args.save_metrics))

    # create trainer
    trainer = pl.Trainer.from_argparse_args(args, logger=False, checkpoint_callback=False, callbacks=callbacks)

    # test
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()

    # define run specific arguments
    parser.add_argument(
        "--save_predictions", default=None, type=str, help="store all predictions as new dataset at given path"
    )
    parser.add_argument("--save_metrics", default=None, type=str, help="compute metrics on predictions and save as csv")

    # add params from Trainer
    parser = pl.Trainer.add_argparse_args(parser)
    # add params from model
    parser = IcoDenoiseNet.add_model_specific_args(parser)
    # add params from datamodule
    parser = IcoMeshPredictionModule.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
