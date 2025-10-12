from lightning.pytorch.cli import LightningCLI

from module import IcoDenoiseNet
from data_loader import IcoMeshDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.subdivisions", "data.subdivisions")
        parser.link_arguments("model.data_internal_scale", "data.data_internal_scale")
        parser.link_arguments("data.batch_size", "model.batch_size")


def cli_main():
    cli = MyLightningCLI(IcoDenoiseNet, IcoMeshDataModule)


if __name__ == "__main__":
    cli_main()
