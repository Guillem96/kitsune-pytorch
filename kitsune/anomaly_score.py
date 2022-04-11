import json
import math
from pathlib import Path
import logging

import torch

from kitsune import engine
from kitsune.data import FileFormat, build_input_data_pipe
from kitsune.models import Kitsune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_anomaly_scores(
    path: Path,
    scores_dir: Path,
    batch_size: int = 32,
    n_scores_partition: int = 4,
    file_format: FileFormat = FileFormat("csv"),
    model_checkpoint: Path = Path("models/kitsune.pt")
) -> dict:

    scores_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Loading model...")
    model = Kitsune.from_pretrained(model_checkpoint)
    model.to(device)

    logging.info("Building input data pipeline...")
    dp = build_input_data_pipe(root=str(path),
                               batch_size=batch_size,
                               file_format=file_format,
                               shuffle=False)

    logging.info("Running inference...")
    scores = engine.predict(model, dp, device=device)

    logging.info("Serializing scores...")
    scores_chunks = scores.split(math.ceil(len(scores) / n_scores_partition))
    for i, chunk in enumerate(scores_chunks):
        fname = scores_dir / f"scores_{i}.json"
        logging.info(f">> Serializing scores chunk {i} ({fname})...")

        with fname.open("w") as score_f:
            lines = [json.dumps({"score": s}) for s in chunk.cpu().tolist()]
            score_f.write("\n".join(lines))

    return scores_chunks