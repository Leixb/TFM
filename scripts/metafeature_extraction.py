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

import argparse
import json
from typing import Dict, Optional, Any, Generator
import glob

import pandas as pd
import numpy as np
import pymfe.mfe as mfe


def extract_dataset(X: np.ndarray, y: Optional[np.ndarray],
                    *args, **kwargs) -> Dict[str, Any]:
    "Extract meta-features from dataset, extra arguments are passed to MFE"

    extractor = mfe.MFE(*args, **kwargs)

    if y is not None:
        extractor.fit(X, y)
    else:
        extractor.fit(X)

    ft = extractor.extract()

    return dict(zip(ft[0], ft[1]))


def process_all(pattern: str = "data/datasets_processed/*.X") -> \
        Generator[Dict[str, Any], None, None]:
    for file in glob.glob(pattern):
        # trim extension
        file = file[:-2]

        X = pd.read_csv(file + ".X").to_numpy()
        y = pd.read_csv(file + ".y", header=None).to_numpy()

        ft = extract_dataset(X, y)

        ft["name"] = file.split("/")[-1]

        yield ft


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        type=str,
        help="Output file to save the metafeatures",
    )
    args = parser.parse_args()

    data = process_all()

    with open(args.output, "w") as f:
        json.dump(list(data), f, default=str)
