import logging
from time import time
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from kitsune.models import FeatureMapper, Kitsune


def timeit(method):
    def timed(*args, **kwargs):
        ts = time()
        result = method(*args, **kwargs)
        msg = f"--------------------{method.__name__} took {time()-ts:2.4f}s\n"
        logging.info(msg)
        return result

    return timed


def get_dimensions(ds: Iterable[torch.Tensor]):

    n_samples = 0
    for _, batch in enumerate(ds):
        n_features = batch.shape[1]
        n_samples += batch.shape[0]

    logging.info(f"{n_samples} samples, {n_features} features")

    return n_samples, n_features


def build_feature_mapper(
    ds: Iterable[torch.Tensor],
    ds_features: int,
    max_features_per_cluster: int = 10,
    fpath: str = "models/fmapper.pkl",
) -> FeatureMapper:
    fm = FeatureMapper(ds_features, max_features_per_cluster)
    for batch in tqdm(ds):
        fm.partial_fit(batch)

    fm.save(fpath)
    return fm


@torch.inference_mode()
def predict(
    model: Kitsune,
    ds: Iterable[torch.Tensor],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    predictions = []
    model.eval()
    for batch in ds:
        batch = batch.to(device)
        predictions.extend(model.score(batch).cpu().tolist())

    return torch.as_tensor(predictions)


@timeit
def train_single_epoch(
    model: Kitsune,
    ds: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    log_every: int = 60,
    epoch: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Kitsune, dict]:
    """Train one round using all the data.

    Parameters
    ----------
    v : Kitsune
        Kitsune model to train
    ds : Iterable[torch.Tensor]
        Itarable to fetch the processed data. Usually a DataLoader
    optimizer : torch.optim.Optimizer
        PyTorch optimizer to optimize the kitsune parameters
    log_every : int, optional
        Report the loss every n steps (in batches), by default 60
    epoch : int, optional
        Only for log purposes, current epoch, by default 0
    device : torch.device, optional
        Device where the model is located, by default torch.device("cpu")
    """
    model.to(device)
    model.train()
    model.zero_grad()

    running_tail = 0.0
    running_head = 0.0
    losses = {}

    for i, sample in enumerate(ds, start=1):
        sample = sample.to(device)

        optimizer.zero_grad()
        losses = model(sample.float())
        tail_loss = losses["tails_losses"].sum(-1).mean(0)
        head_loss = losses["head_loss"].mean(0)
        tail_loss.backward()
        head_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        running_tail += tail_loss.item()
        running_head += head_loss.item()

        if i % log_every == 0:
            tail_loss_mean = running_tail / i
            head_loss_mean = running_head / i
            logging.info(
                f"Epoch [{epoch}]  "
                f"tail losses {tail_loss_mean:.5f}  "
                f"head loss: {head_loss_mean:.5f}"
            )
            losses = {
                "epoch": epoch,
                "tail_losses": tail_loss_mean,
                "head_loss": head_loss_mean,
            }

    return model, losses
