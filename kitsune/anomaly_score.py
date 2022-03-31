import json
import math
from pathlib import Path

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
    file_format: FileFormat = "csv",
    model_checkpoint: Path = Path("models/kitsune.pt")
) -> None:

    scores_dir.mkdir(exist_ok=True, parents=True)

    print("🦊 Loading model...")
    model = Kitsune.from_pretrained(model_checkpoint)
    model.to(device)

    print("🦊 Building input data pipeline...")
    dp = build_input_data_pipe(root=str(path),
                               batch_size=batch_size,
                               file_format=file_format,
                               shuffle=False)

    print("🦊 Running inference...")
    scores = engine.predict(model, dp, device=device)

    print("🦊 Serializing scores...")
    scores_chunks = scores.split(math.ceil(len(scores) / n_scores_partition))
    for i, chunk in enumerate(scores_chunks):
        fname = scores_dir / f"scores_{i}.json"
        print(f">> Serializing scores chunk {i} ({fname})...")

        with fname.open("w") as score_f:
            lines = [json.dumps({"score": s}) for s in chunk.cpu().tolist()]
            score_f.write("\n".join(lines))
