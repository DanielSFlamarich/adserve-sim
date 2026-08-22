# src/adserve_sim/data/download.py

"""Fetch the Avazu CTR dataset and prepare a laptop-sized Parquet sample.

The raw competition file is roughly 6 GB / 40M rows, which is slower to iterate
on than it is informative. This module takes a per-hour stratified sample so the
temporal shape of the traffic (the daily volume curve the pacing controller
will later have to track) is preserved at a fraction of the size.

Raw data is never committed; see ``.gitignore``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from adserve_sim.data.schema import (
    RAW_DTYPES,
    TIMESTAMP,
    parse_hour,
    validate_columns,
)

logger = logging.getLogger(__name__)

#: kaggle competition slug for the raw dataset.
COMPETITION: str = "avazu-ctr-prediction"

#: filename of the training split inside the competition archive.
RAW_FILENAME: str = "train.gz"

DATA_DIR: Path = Path("data")
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

#: default number of rows to retain in the prepared sample.
DEFAULT_SAMPLE_ROWS: int = 3_000_000

#: chunk size for the streaming read of the raw CSV.
CHUNK_ROWS: int = 1_000_000

_MANUAL_INSTRUCTIONS = f"""
Raw file not found.

To fetch it manually:
  1. Accept the competition rules at
     https://www.kaggle.com/competitions/{COMPETITION}/rules
     Downloads are refused until the rules are accepted, even with valid
     credentials, and listing files still works, so this failure is easy to
     mistake for an auth problem.
  2. Download the data and place `{RAW_FILENAME}` at:
     {{target}}

The Kaggle CLI route needs an API token at ~/.kaggle/access_token; see
https://www.kaggle.com/docs/api
"""


def file_digest(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read incrementally.

    Args:
        path: File to hash.
        chunk_bytes: Read buffer size.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def fetch_raw(raw_dir: Path = RAW_DIR) -> Path:
    """Ensure the raw Avazu training file is present locally.

    Tries the Kaggle CLI if the file is absent. Kaggle requires accepting the
    competition rules interactively, so this cannot be made fully unattended;
    on failure the caller is told how to place the file by hand.

    Args:
        raw_dir: Directory holding raw downloads.

    Returns:
        Path to the raw training file.

    Raises:
        FileNotFoundError: If the file is absent and cannot be fetched.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_FILENAME

    if target.exists():
        logger.info("raw file already present at %s", target)
        return target

    logger.info("raw file missing; attempting Kaggle download")
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-f",
                RAW_FILENAME,
                "-p",
                str(raw_dir),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        # the CLI ran and refused, message says why: usually unaccepted
        # competition rules, and capture_output means nobody sees it unless we
        # put it back.
        detail = exc.stderr.decode(errors="replace").strip()
        raise FileNotFoundError(
            f"{_MANUAL_INSTRUCTIONS.format(target=target)}\nKaggle CLI reported:\n{detail}"
        ) from exc
    except FileNotFoundError as exc:
        # the CLI is not installed at all.
        raise FileNotFoundError(
            f"{_MANUAL_INSTRUCTIONS.format(target=target)}\n"
            "(@_@) The kaggle CLI was not found; install it with: uv add --dev kaggle"
        ) from exc

    # CLI may deliver either the file itself or a zip wrapping it.
    archive = raw_dir / f"{RAW_FILENAME}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        archive.unlink()

    if not target.exists():
        raise FileNotFoundError(_MANUAL_INSTRUCTIONS.format(target=target))

    logger.info("downloaded %s (sha256=%s)", target, file_digest(target)[:16])
    return target


def prepare_sample(
    raw_path: Path,
    output_path: Path,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    seed: int = 42,
) -> pd.DataFrame:
    """Read the raw CSV in chunks and write a per-hour stratified Parquet sample.

    Sampling is proportional within each hour, so the hourly volume curve of the
    original traffic is preserved up to a constant factor. Sampling uniformly at
    random across the whole file would preserve it in expectation too, but
    stratifying makes the guarantee exact per hour, which matters because later
    steps split on day boundaries.

    Args:
        raw_path: Path to the raw gzipped CSV.
        output_path: Destination Parquet path.
        sample_rows: Approximate number of rows to retain.
        seed: Seed for the sampling draw.

    Returns:
        The prepared frame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[pd.DataFrame] = []
    total_rows = 0

    reader = pd.read_csv(
        raw_path,
        dtype=RAW_DTYPES,
        chunksize=CHUNK_ROWS,
        compression="infer",
    )

    for index, chunk in enumerate(reader):
        if index == 0:
            validate_columns(chunk)
        chunks.append(chunk)
        total_rows += len(chunk)
        logger.info("read chunk %d (%d rows so far)", index, total_rows)

    frame = pd.concat(chunks, ignore_index=True)
    del chunks

    frame[TIMESTAMP] = parse_hour(frame["hour"])

    fraction = min(1.0, sample_rows / len(frame))
    if fraction < 1.0:
        frame = (
            frame.groupby(TIMESTAMP, group_keys=False, observed=True)
            .sample(frac=fraction, random_state=seed)
            .reset_index(drop=True)
        )

    frame = frame.sort_values(TIMESTAMP).reset_index(drop=True)
    frame.to_parquet(output_path, index=False)

    logger.info(
        "wrote %d rows to %s (base CTR %.4f)",
        len(frame),
        output_path,
        frame["click"].mean(),
    )
    return frame


def main() -> None:
    """Command-line entry point: fetch the raw file and write the sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="approximate number of rows to retain",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "avazu_sample.parquet",
        help="destination Parquet path",
    )
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_path = fetch_raw()
    prepare_sample(
        raw_path=raw_path,
        output_path=args.output,
        sample_rows=args.sample_rows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
