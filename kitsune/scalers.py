import json
import os
import pickle
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import torch
from torchdata.datapipes.iter import IterDataPipe
from tqdm import tqdm

DataPipeIterTensors = Union[IterDataPipe[torch.Tensor], Iterable[torch.Tensor]]


class BatchTorchMinMaxScaler:
    """MinMax Scaler for a batched DataPipeline of Torch Tensors"""

    def __init__(
        self,
        feature_range: Tuple[int, int] = (0, 1),
        dim: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """

        Args:
            feature_range (Tuple[int, int], optional): _description_. Defaults to (0, 1).
            dim (Optional[int], optional): Read from the first batch data if not provided.
            batch_size (Optional[int], optional): Read from the first batch data if not provided.
        """

        self.feature_range = feature_range
        self.dim = dim
        self.batch_size = batch_size
        self.x_min: torch.tensor
        self.x_max: torch.tensor
        self.xmin_batch: torch.tensor
        self.xmaxmxmin_batch: torch.tensor

    def _init_with_data(self, x_batch: torch.Tensor):

        self.batch_size, self.dim = x_batch.shape
        self.x_min = torch.tensor([float("inf") for n in range(self.dim)])
        self.x_max = torch.tensor([float("-inf") for n in range(self.dim)])

        return

    def _create_batch_tensors(self):
        """create a tensor with size of batch filled with duplicates of xmnin and xmaxmin"""

        self.xmin_batch = torch.vstack([self.x_min for bb in range(self.batch_size)])
        self.xmaxmxmin_batch = (
            torch.vstack([self.x_max for bb in range(self.batch_size)])
            - self.xmin_batch
        )

        return

    def _setup_fit(self, batch: torch.Tensor):
        self._init_with_data(batch)
        self._create_batch_tensors()
        return

    def fit(self, X: DataPipeIterTensors):
        """Fit scaler on X

        Args:
            X (DataPipeIterTensors)
        """

        if self.dim is None or self.batch_size is None:
            self._setup_fit(next(iter(X)))

        for _, batch in tqdm(enumerate(X)):
            x_min = torch.min(batch, axis=0).values
            x_max = torch.max(batch, axis=0).values

            self.x_min = torch.minimum(self.x_min, x_min)
            self.x_max = torch.maximum(self.x_max, x_max)

        return

    def _setup_transform(self, X: DataPipeIterTensors):

        self.batch_size, self.dim = next(iter(X)).shape
        self._create_batch_tensors()

        return

    def transform(self, X: DataPipeIterTensors) -> DataPipeIterTensors:
        """Scale X with pre-fit scaler

        Args:
            X (DataPipeIterTensors):

        Returns:
            DataPipeIterTensors:
        """

        self._setup_transform(X)

        if isinstance(X, IterDataPipe):
            return X.map(lambda x: self._scale_batch(x))

        elif isinstance(X, list):
            return list(map(lambda x: self._scale_batch(x), X))

    def _scale_batch(self, batch: torch.Tensor):
        xp = (batch - self.xmin_batch) / self.xmaxmxmin_batch
        xp = (
            xp * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        )
        xp = torch.nan_to_num(xp)  # for columns with constant values
        return xp

    def save(self, path: Path, file: str = "scaler.pkl"):
        """save as pickle"""
        self.save_pickle(path, file)
        return

    def save_pickle(self, path: Path, file: str = "scaler.pkl"):
        """save scaler to a pickle file

        Args:
            path (Path):
            file (str, optional): Defaults to "scaler.pkl".
        """
        if os.path.exists(path) is False:
            os.makedirs(path)

        with open(path / file, "wb") as f:
            pickle.dump(self, f)

        return

    def save_json(self, path: Path, file: str = "scaler.json"):
        """save scaler parameters to a json file, for maximum  compatibility

        Args:
            path (Path):
            file (str, optional): Defaults to "scaler.json".
        """
        parameters = {
            "feature_range": self.feature_range,
            "dim": self.dim,
            "batch_size": self.batch_size,
            "x_min": self.x_min.tolist(),
            "x_max": self.x_max.tolist(),
        }

        with open(path / file, "w") as f:
            json.dump(parameters, f)

        return

    @classmethod
    def load(cls, path: Path, file: str = "scaler.pkl"):
        return BatchTorchMinMaxScaler.load_pickle(path, file)

    @classmethod
    def load_pickle(cls, path: Path, file: str = "scaler.pkl"):
        """load scaler from pickle

        Args:
            path (Path):
            file (str, optional):. Defaults to "scaler.pkl".

        Returns:
            scaler
        """
        with open(path / file, "rb") as f:
            scaler: BatchTorchMinMaxScaler = pickle.load(f)

        return scaler

    @classmethod
    def load_json(cls, path: Path, file: str = "scaler.json"):
        """instantiate a Scaler from parameters loaded from a json file

        Args:
            path (Path):
            file (str, optional): Defaults to "scaler.json".

        Returns:
            _type_:
        """
        with open(path / file, "r") as f:
            parameters = json.load(f)

        scaler: BatchTorchMinMaxScaler = BatchTorchMinMaxScaler(
            parameters["feature_range"], parameters["dim"], parameters["batch_size"]
        )
        scaler.x_min = torch.FloatTensor(parameters["x_min"])
        scaler.x_max = torch.FloatTensor(parameters["x_max"])

        return scaler
