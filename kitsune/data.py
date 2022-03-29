import enum
from itertools import zip_longest
from typing import IO, Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe

ParquetRow = Dict[str, Any]
ParquetParserReturnType = Union[ParquetRow, Tuple[str, Any]]


class FileFormat(enum.Enum):
    csv: str = "csv"
    parquet: str = "parquet"


@functional_datapipe("parse_parquet")
class ParquetParserIterDataPipe(IterDataPipe[ParquetParserReturnType]):

    def __init__(self,
                 source_datapipe: IterDataPipe[Tuple[str, IO]],
                 *,
                 return_path: bool = False,
                 engine: str = "auto",
                 columns: Optional[List[str]] = None) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_path = return_path
        self.engine = engine
        self.columns = columns

    def __iter__(self) -> Iterator[ParquetParserReturnType]:
        for path, io_file in self.source_datapipe:
            parquet_df = pd.read_parquet(io_file,
                                         engine=self.engine,
                                         columns=self.columns)

            it = (o[1].to_dict() for o in parquet_df.iterrows())
            if self.return_path:
                yield from zip_longest([path], it)
            else:
                yield from it


def build_input_data_pipe(
        root: str,
        batch_size: int,
        shuffle: bool = True,
        file_format: FileFormat = FileFormat.csv) -> IterDataPipe[torch.Tensor]:

    file_filter = lambda fname: fname.endswith(file_format.value)
    dp = FileLister(root=root, recursive=True).filter(file_filter)
    dp = FileOpener(dp, mode="rb")

    if file_format == FileFormat.parquet:
        dp = dp.parse_parquet(return_path=False, columns=["features"])
        dp = dp.map(_to_dense_vector)
    else:
        dp = dp.parse_csv(return_path=False)
        dp = dp.map(lambda x: torch.as_tensor([float(o) for o in x]))

    if shuffle:
        dp = dp.shuffle()

    dp = dp.batch(batch_size=batch_size, drop_last=False)
    return dp.collate(torch.stack)


def _to_dense_vector(row: ParquetRow) -> torch.Tensor:
    sample_features = row["features"]
    if sample_features["indices"] is not None:
        features = torch.zeros((int(sample_features["size"]),)).float()
        indices = torch.as_tensor(sample_features["indices"].copy()).long()
        features[indices] = torch.as_tensor(
            sample_features["values"].copy()).float()
    else:
        features = torch.as_tensor(sample_features["values"].copy()).float()
    return features


if __name__ == "__main__":
    # parquet_dp = FileLister(root="data/parquet")
    dp = build_input_data_pipe(root="data/kitnet-test",
                               batch_size=16,
                               file_format=FileFormat.csv)
    for features in dp:
        print(features.size())
        break
