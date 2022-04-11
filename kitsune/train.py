import random
from pathlib import Path
import logging
from typing import Optional
import json

import numpy as np
import torch

from kitsune.data import FileFormat, build_input_data_pipe
from kitsune.engine import get_dimensions, build_feature_mapper, train_single_epoch
from kitsune.models import Kitsune, FeatureMapper
from kitsune.scalers import BatchTorchMinMaxScaler

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    input_path: Path,
    batch_size: int = 32,
    file_format: FileFormat = FileFormat("csv"),
    compression_rate: float = 0.6,
    checkpoint_dir: Path = Path("models/"),
    max_features_per_cluster:int=10,
    epochs:int=1,
    log_freq:int=5,
    sgd_lr:float=1e-3,
    sgd_momentum=0.9,
    n_samples:Optional[int]=None,
    n_features:Optional[int]=None,
    pretrained_mapper:Optional[str]=None,
    prefit_scaler:Optional[str]=None,
    seed:int=0
    ) -> dict:
    """ train a Kitsune model

        Note that feature_mapper and scaler may be batch dependent, so if you change the batch, you should re calculate them

    Args:
        input_path (Path): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        file_format (FileFormat, optional): _description_. Defaults to "csv".
        compression_rate (float, optional): _description_. Defaults to 0.6.
        checkpoint_dir (Path, optional): _description_. Defaults to Path("models/").
        max_features_per_cluster (int, optional): _description_. Defaults to 10.
        log_freq (int, optional): How many log prints per EPOCH. Defaults to 5.
        sgd_lr (float, optional): _description_. Defaults to 1e-3.
        sgd_momentum (float, optional): _description_. Defaults to 0.9.
        n_samples (Optional[int], optional): _description_. Defaults to None.
        n_features (Optional[int], optional): _description_. Defaults to None.
        pretrained_mapper (Optional[str], optional): _description_. Defaults to None.
        prefit_scaler (Optional[str], optional): _description_. Defaults to None.

    Raises:
        Exception: If batch does not cover dataset size
    """  

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dp = build_input_data_pipe(str(input_path),
                               batch_size=batch_size,
                               shuffle=True,
                               file_format=file_format)


    if n_samples is None or n_features is None:
        logging.info("🦊 Getting data dimensions ...")
        n_samples, n_features = get_dimensions(dp)

    if n_samples % batch_size != 0:
        raise Exception(f' n_samples {n_samples} modulo batch_size {batch_size} should be zero but is not, choose a different batch size')

    if pretrained_mapper is None:
        logging.info("🦊 Training feature mapper ...")

        feature_mapper = build_feature_mapper(dp,
                                            ds_features=n_features,
                                            max_features_per_cluster=max_features_per_cluster)

        logging.info(f"Feature mapping: {feature_mapper.clusters_}")
    else:
        feature_mapper = FeatureMapper.load(pretrained_mapper)

    if prefit_scaler is None:
        logging.info('🦊 Fitting scaler ...')
        scaler = BatchTorchMinMaxScaler()
        scaler.fit(dp)
        dp = scaler.transform(dp)
    else:
        scaler : BatchTorchMinMaxScaler = BatchTorchMinMaxScaler.load(Path(prefit_scaler).parent)
        dp = scaler.transform(dp)

    model = Kitsune(feature_mapper=feature_mapper,
                    compression_rate=compression_rate)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=sgd_momentum)

    logging.info("🦊 Training Kitsune ensemble ...")
    log_every = n_samples / (log_freq * batch_size) 

    for epoch in range(1, epochs+1):
        model, losses = train_single_epoch(model=model, ds=dp, optimizer=optimizer, device=device, log_every=log_every, epoch=epoch)

    logging.info(f"🦊 Serializing the model to {checkpoint_dir / 'kitsune.pt'} ...")
    scaler.save(checkpoint_dir)
    model.save(checkpoint_dir / "kitsune.pt")
    with open (checkpoint_dir / 'losses.json', 'wt') as f:
        json.dump(losses, f)

    return losses