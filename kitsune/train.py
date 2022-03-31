import random
from pathlib import Path
from typing import Iterator, TypeVar, Union

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from kitsune.data import FileFormat, build_input_data_pipe
from kitsune.engine import build_feature_mapper, predict, train_single_epoch
from kitsune.models import Kitsune

T = TypeVar("T")
device = torch.device("cuda" if torch.cuda.is_available() "cpu")

@functional_datapipe("skip")
class SkipIterDataPipe(IterDataPipe[T]):

    def __init__(self, source_dp: IterDataPipe[T], *, n: int) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.n = n

    def __iter__(self) -> Iterator[T]:
        for i, o in enumerate(self.source_dp):
            if i >= self.n:
                yield o


@functional_datapipe("take")
class TakeIterDataPipe(IterDataPipe[T]):

    def __init__(self, source_dp: IterDataPipe[T], *, n: int) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.n = n

    def __iter__(self) -> Iterator[T]:
        for i, o in enumerate(self.source_dp):
            if i < self.n:
                yield o
            else:
                break


@functional_datapipe("minmax_scale")
class MinMaxScalerIterDataPipe(IterDataPipe[torch.Tensor]):

    def __init__(self, source_dp: IterDataPipe[torch.Tensor],
                 n_samples: int) -> None:
        super().__init__()
        self.source_dp = source_dp
        self._min_max_scaler = MinMaxScaler()
        self._fitted = False
        self.n_samples = n_samples

    def fit(self) -> None:
        # use_to_fit, self.source_dp = self.source_dp.fork(2, buffer_size=10_000)
        print("🦊 Fitting min-max scaler...")
        viewed_samples = 0
        for batch in self.source_dp:
            self._min_max_scaler.partial_fit(batch.numpy())
            viewed_samples += len(batch)
            if viewed_samples >= self.n_samples:
                break
        self._fitted = True

    def __iter__(self) -> Iterator[torch.Tensor]:
        if not self._fitted:
            self.fit()

        for batch in self.source_dp:
            transformed = self._min_max_scaler.transform(batch.numpy())
            yield torch.as_tensor(transformed)


def train(input_path: Path,
          batch_size: int = 32,
          file_format: FileFormat = "csv",
          compression_rate: float = 0.6,
          checkpoint_dir: Path = Path("models/"),
          is_scaled: bool = False) -> None:

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    dp = build_input_data_pipe(str(input_path),
                               batch_size=batch_size,
                               shuffle=False,
                               file_format=file_format)

    if not is_scaled:
        dp = dp.minmax_scale(n_samples=55000)

    train_dp = dp.take(n=55000 // batch_size).shuffle()
    test_dp = dp.skip(n=55000 // batch_size)

    print("🦊 Training feature mapper ...")
    feature_mapper = build_feature_mapper(dp,
                                          ds_features=115,
                                          max_features_per_cluster=10)

    model = Kitsune(feature_mapper=feature_mapper,
                    compression_rate=compression_rate)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    train_single_epoch(kitsune=model, 
                       ds=train_dp, 
                       optimizer=optimizer,
                       device=device)

    model.save(checkpoint_dir / "kitsune.pt")

    print("🦊 Computing train anomaly scores...")
    train_rmse = predict(model, train_dp)

    print("🦊 Computing test anomaly scores...")
    test_rmse = predict(model, test_dp)

    from scipy.stats import norm
    log_train_rmse = train_rmse.log()
    log_test_rmse = test_rmse.log()

    log_rmses = torch.cat([log_train_rmse, log_test_rmse], dim=-1)
    log_probs = norm.logpdf(log_rmses,
                            loc=log_train_rmse.mean(),
                            scale=log_train_rmse.std())

    # plot the RMSE anomaly scores
    print("Plotting results")
    import pandas as pd
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 5))
    timestamps = pd.read_csv("data/mirai3_ts.csv", header=None).values
    print(log_rmses.shape, timestamps.shape, log_test_rmse.shape,
          timestamps[55000:].shape)

    plt.scatter(timestamps[55000:],
                log_rmses[55000:],
                s=0.1,
                c=log_probs[55000:],
                cmap='RdYlGn')
    plt.title("Anomaly Scores from KitNET's Execution Phase")
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("Time elapsed [min]")
    plt.annotate(
        'Mirai C&C channel opened [Telnet]',
        xy=(timestamps[71662], log_rmses[71662]),
        xytext=(timestamps[58000], 1),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )
    plt.annotate(
        'Mirai Bot Activated\nMirai scans network for vulnerable devices',
        xy=(timestamps[72662], 1),
        xytext=(timestamps[55000], 5),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )
    figbar = plt.colorbar()
    figbar.ax.set_ylabel('Log Probability\n ', rotation=270)

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps[55000:], log_rmses[55000:], ".")
    plt.axhline(y=log_train_rmse.mean() + log_train_rmse.std() * 3,
                color='r',
                linestyle='-')
    plt.show()
