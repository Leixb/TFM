#!/usr/bin/env python3
"""
Extract metafeatures from a given dataset using pymfe
=====================================================

This should be done after all the preprocessing is done from Julia,
so the dataset is ready to be used as it would be in a real scenario,
this means, that we have to run this from Julia, or at least, we need
to run Julia and save the intermediate results to a file which can be
read here.

This can be used as both a command line tool or a module, which enables
it to be used from within Julia. This is the ideal scenario, but in reality
PyCall in Julia can have issues interfacing with Python modules if they are
not installed using `Conda.jl`.

See the `scipts/extract_metafeatures.jl` file for the implementation in Julia
of the dataset preprocessing and saving location.
"""

from typing import Dict, Optional, Any, Generator
import argparse
import glob
import json
import sys

import pandas as pd
import numpy as np
import pymfe.mfe as mfe  # type: ignore


def extract_dataset(X: np.ndarray, y: Optional[np.ndarray],
                    extract_args: Dict[str, Any] = {
                        "verbose": 1, "suppress_warnings": True},
                    *args, **kwargs) -> Dict[str, Any]:
    "Extract meta-features from dataset, extra arguments are passed to MFE"

    extractor = mfe.MFE(*args, **kwargs)

    if y is not None:
        extractor.fit(X, y)
    else:
        extractor.fit(X)

    ft = extractor.extract(**extract_args)

    return dict(zip(ft[0], ft[1]))


def process_all(pattern: str = "data/datasets_processed/*.X",
                *args, **kwargs) -> Generator[Dict[str, Any], None, None]:
    for file in glob.glob(pattern):
        # trim extension
        file = file[:-2]

        # WARN: skip MNIST dataset since it is too big, should be computed
        # separately with a subset of the data
        if file.endswith("MNIST"):
            print(f"Skipping {file}", file=sys.stderr)
            continue

        print(f"Processing {file}", file=sys.stderr)

        X = pd.read_csv(file + ".X").to_numpy()
        y = pd.read_csv(file + ".y", header=None).to_numpy()

        ft = extract_dataset(X, y, *args, **kwargs)

        ft["name"] = file.split("/")[-1]

        yield ft


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        help="Output file to save the metafeatures",
        default=sys.stdout
    )
    parser.add_argument(
        "--features",
        nargs="+",
        help="Groups of features to extract, if not given, default are extracted",
        default=["general", "statistical", "model-based", "landmarking"]
    )
    args = parser.parse_args()

    with args.output as f:
        for ft in process_all(groups=args.features):
            json.dump(ft, f, default=str)
            f.write("\n")
            f.flush()
