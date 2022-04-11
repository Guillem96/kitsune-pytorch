import pickle
from typing import Iterable, Tuple, Union
from pathlib import Path
import os

from tqdm import tqdm
from torchdata.datapipes.iter import IterDataPipe
import torch


class BatchTorchMinMaxScaler:
    """MinMax Scaler for a batched DataPipeline of Torch Tensors"""

    def __init__(self, feature_range: Tuple[int, int] = (0, 1)) -> None:

        self.feature_range = feature_range
        self.dim: int
        self.batch_size: int

    def __init__with_data__(self, x_batch: torch.Tensor):

        self.batch_size, self.dim = x_batch.shape
        self.x_min = torch.tensor([float("inf") for n in range(self.dim)])
        self.x_max = torch.tensor([float("-inf") for n in range(self.dim)])

        return

    def fit(self, X: Union[Iterable[torch.Tensor], IterDataPipe[torch.Tensor]]):

        for ib, batch in tqdm(enumerate(X)):

            if ib == 0:
                self.__init__with_data__(batch)

            x_min_batch = torch.min(batch, axis=0).values
            x_max_batch = torch.max(batch, axis=0).values

            self.x_min = torch.minimum(self.x_min, x_min_batch)
            self.x_max = torch.maximum(self.x_max, x_max_batch)

        self.xmin_batch = torch.vstack([self.x_min for bb in range(self.batch_size)])
        self.xmaxmxmin_batch = (
            torch.vstack([self.x_max for bb in range(self.batch_size)])
            - self.xmin_batch
        )

    def transform(self, X: IterDataPipe[torch.Tensor]) -> IterDataPipe[torch.Tensor]:
        def scale_batch(x):
            xp = (x - self.xmin_batch) / self.xmaxmxmin_batch
            xp = (
                xp * (self.feature_range[1] - self.feature_range[0])
                + self.feature_range[0]
            )
            xp = torch.nan_to_num(xp)  # for columns with constant values
            return xp

        return X.map(lambda x: scale_batch(x))

    def save(self, path: Path, file: str = "scaler.pkl") -> None:

        if os.path.exists(path) is False:
            os.mkdir(path)

        with open(path / file, "wb") as f:
            pickle.dump(self, f)

        return

    @classmethod
    def load(cls, path: Path, file: str = "scaler.pkl"):

        with open(path / file, "rb") as f:
            scaler = pickle.load(f)

        return scaler
