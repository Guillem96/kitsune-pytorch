from typing import Iterable

import torch
import torch.nn as nn

from kitsune.models import FeatureMapper, Kitsune


def build_feature_mapper(ds: Iterable[torch.Tensor],
                         ds_features: int,
                         max_features_per_cluster: int = 10) -> FeatureMapper:
    fm = FeatureMapper(ds_features, max_features_per_cluster)
    for batch in ds:
        fm.parital_fit(batch)
    return fm


@torch.inference_mode()
def predict(model: Kitsune, 
            ds: Iterable[torch.Tensor], 
            device: torch.device = torch.device("cpu")) -> torch.Tensor:
    predictions = []
    model.eval()
    for batch in ds:
        batch = batch.to(device)
        predictions.extend(model.score(batch).cpu().tolist())

    return torch.as_tensor(predictions)


def train_single_epoch(
    model: Kitsune,
    ds: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    log_every: int = 60,
    epoch: int = 0,
    device: torch.device = torch.device("cpu")) -> None:
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
        Report the loss every n steps, by default 30
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
            print(f"🦊 Epoch [{epoch}]  "
                  f"tail losses {tail_loss_mean:.5f}  "
                  f"head loss: {head_loss_mean:.5f}")
