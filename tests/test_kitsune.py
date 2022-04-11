from operator import mod
from pathlib import Path

import numpy as np

from kitsune.anomaly_score import compute_anomaly_scores
from kitsune.data import FileFormat, build_input_data_pipe
from kitsune.scalers import BatchTorchMinMaxScaler
from kitsune.train import train


def test_input_data_pipe():

    dp = build_input_data_pipe(
        "tests/mirai100.csv", batch_size=10, shuffle=True, file_format=FileFormat("csv")
    )

    assert True


def test_scaler(tmp_path: Path):

    dp = build_input_data_pipe(
        "tests/mirai100.csv", batch_size=10, shuffle=True, file_format=FileFormat("csv")
    )

    scaler = BatchTorchMinMaxScaler()
    scaler.fit(dp)
    dp = scaler.transform(dp)

    assert int(scaler.x_max.mean().numpy().round()) == 9642608427008
    assert int(scaler.x_min.mean().numpy().round()) == 222673936
    assert int(scaler.x_min.min().numpy().round()) == -40

    data = np.vstack(np.array([batch.numpy() for batch in list(dp)]))

    assert (round(np.mean(data), 3) > 0.188) and (round(np.mean(data), 3) < 0.200)
    assert (round(np.std(data), 3) > 0.272) and (round(np.mean(data), 3) < 0.274)

    scaler.save(tmp_path)

    dp = build_input_data_pipe(
        "tests/mirai100.csv", batch_size=10, shuffle=True, file_format=FileFormat("csv")
    )
    scaler: BatchTorchMinMaxScaler = scaler.load(tmp_path)
    dp = scaler.transform(dp)

    assert int(scaler.x_max.mean().numpy().round()) == 9642608427008
    assert int(scaler.x_min.mean().numpy().round()) == 222673936
    assert int(scaler.x_min.min().numpy().round()) == -40

    data = np.vstack(np.array([batch.numpy() for batch in list(dp)]))

    assert (round(np.mean(data), 3) > 0.188) and (round(np.mean(data), 3) < 0.200)
    assert (round(np.std(data), 3) > 0.272) and (round(np.mean(data), 3) < 0.274)


def test_training(tmp_path: Path):

    losses = train(
        input_path=Path("tests/mirai100.csv"),
        batch_size=10,
        checkpoint_dir=tmp_path,
        epochs=2,
        seed=0,
    )

    assert losses["epoch"] == 2
    assert losses["tail_losses"] > 6.413 and losses["tail_losses"] < 6.420
    assert losses["head_loss"] >= 0.124 and losses["head_loss"] < 0.125


def test_scoring(tmp_path: Path):

    train(
        input_path=Path("tests/mirai100.csv"),
        batch_size=10,
        checkpoint_dir=tmp_path,
        epochs=2,
        seed=0,
    )

    scores = compute_anomaly_scores(
        path=Path("tests/mirai100.csv"),
        scores_dir=tmp_path / "scores",
        batch_size=10,
        n_scores_partition=4,
        model_checkpoint=tmp_path / "kitsune.pt",
    )

    assert scores[0][0].numpy() > 549401151 and scores[0][0].numpy() < 549401153
    assert scores[1][4].numpy() > 545751743 and scores[1][4].numpy() < 545751745
    assert scores[0].mean().numpy() > 1.4297e12 and scores[0].mean().numpy() < 1.4299e12
    assert scores[0].std().numpy() > 3.1910e12 and scores[0].std().numpy() < 3.1912e12


if __name__ == "__main__":

    test_scaler(Path("./local"))

    test_training(Path("./local"))

    test_scoring(Path("./local"))
