import random
from pathlib import Path

import numpy as np
import torch

from kitsune.data import FileFormat, build_input_data_pipe
from kitsune.engine import build_feature_mapper, train_single_epoch
from kitsune.models import Kitsune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    input_path: Path,
    batch_size: int = 32,
    file_format: FileFormat = "csv",
    compression_rate: float = 0.6,
    checkpoint_dir: Path = Path("models/")) -> None:

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    dp = build_input_data_pipe(str(input_path),
                               batch_size=batch_size,
                               shuffle=True,
                               file_format=file_format)

    print("🦊 Training feature mapper ...")
    feature_mapper = build_feature_mapper(dp,
                                          ds_features=115,
                                          max_features_per_cluster=10)

    model = Kitsune(feature_mapper=feature_mapper,
                    compression_rate=compression_rate)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    print("🦊 Training Kitsune ensemble ...")
    train_single_epoch(model=model, ds=dp, optimizer=optimizer, device=device)

    print(f"🦊 Serializing the model to {checkpoint_dir / 'kitsune.pt'} ...")
    model.save(checkpoint_dir / "kitsune.pt")
