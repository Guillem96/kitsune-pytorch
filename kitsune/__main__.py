import typer

from kitsune.train import train

kitsune_cli = typer.Typer(name="Kitsune Network")
kitsune_cli.command(name="train")(train)


@kitsune_cli.callback()
def main() -> None:
    """Kitsune anomaly detector Command Line Interface."""


if __name__ == "__main__":
    kitsune_cli()
